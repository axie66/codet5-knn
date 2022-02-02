import os
import time
import math
import logging
from tqdm import tqdm
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from data.conala.evaluation.compute_eval_metrics import compute_metric
from dataset import get_elapse_time, load_conala_dataset, preprocess_batch_conala
from config import add_args, add_knn_args, parse_args, set_seed
from knn import KNN_Dstore

from model import T5KNN, T5ForConditionalGeneration, T5AttnKNN
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

logger = logging.getLogger(__name__)
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print(device)

try:
    import wandb
except ImportError:
    logger.warn('Unable to import wandb')


def train_epoch(args, model, train_dataloader, tokenizer, optimizer, scheduler, 
                cur_epoch, global_step):
    bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
    nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
    model.train()

    epoch_stats = {}

    for step, batch in enumerate(bar):
        source_ids = batch['source']['input_ids'].to(device)
        target_ids = batch['target']['input_ids'].to(device)
        source_mask = batch['source']['attention_mask'].to(device)
        target_mask = batch['target']['attention_mask'].to(device)

        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                        labels=target_ids, decoder_attention_mask=target_mask)
        loss = outputs.loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()

        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if nb_tr_steps % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
            bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

    epoch_stats['train_loss'] = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
    return epoch_stats, global_step


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.batch_size,
        num_workers=4 if cuda else 0, collate_fn=preprocess_batch_conala)

    model.eval()
    pred_ids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch['source']['input_ids'].to(device)
        source_mask = batch['source']['attention_mask'].to(device)
        with torch.no_grad():
            preds = model.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    num_beams=args.beam_size,
                                    early_stopping=False,
                                    max_length=args.max_target_length)
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [{
        'token': torch.tensor(ids, requires_grad=False),
        'str': tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    } for ids in pred_ids]

    # output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    # gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    # src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    # dev_accs = []
    # with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
    #     for pred_nl, gold in zip(pred_nls, eval_examples):
    #         dev_accs.append(pred_nl.strip() == gold.target.strip())
    #         f.write(pred_nl.strip() + '\n')
    #         f1.write(gold.target.strip() + '\n')
    #         f2.write(gold.source.strip() + '\n')

    metrics, sampled_texts = compute_metric(pred_nls, 'conala', split=split_tag,
        tokenizer=tokenizer, args=args, return_data=True)

    metrics['em'] = metrics.pop('exact_match')

    logger.info("***** Eval results *****")
    for key in sorted(metrics.keys()):
        try:
            logger.info("  %s = %s", key, str(round(metrics[key], 4)))
        except TypeError:
            pass

    return metrics, sampled_texts


def main():
    global sampled_texts

    parser = add_knn_args(add_args(argparse.ArgumentParser()))
    args = parse_args(parser)

    logger.info(args)
    t0 = time.time()

    set_seed(args)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    if getattr(args, 'k', None):
        model = T5KNN.from_pretrained('Salesforce/codet5-base')
        model.set_knn_dstore(KNN_Dstore(args, vocab_size=tokenizer.vocab_size, 
            pad_idx=tokenizer.pad_token_id))
    else:
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    if os.path.isfile(args.model_name_or_path):
        print('Loaded pretrained weights from', args.model_name_or_path)
        model.load_state_dict(torch.load(args.model_name_or_path))

    print("Using model of type", type(model))
    model.to(device)

    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.wandb:
        wandb.init('knn-code-gen', config=vars(args))

    if args.do_train:
        # Prepare data loaders
        train_data, valid_data, test_data = load_conala_dataset(args, tokenizer)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=preprocess_batch_conala, num_workers=4 if cuda else 0)
        valid_dataloader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size,
                                      collate_fn=preprocess_batch_conala, num_workers=4 if cuda else 0)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=preprocess_batch_conala, num_workers=4 if cuda else 0)

        # Prepare optimizer and schedule (linear warmup and decay)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            epoch_stats, global_step = train_epoch(
                args, model, train_dataloader, tokenizer, optimizer, scheduler, 
                cur_epoch, global_step
            )

            if args.do_eval:
                result, sampled_texts = eval_bleu_epoch(args, valid_data, valid_data.data, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                dev_bleu, dev_em = result['bleu'], result['em']
                dev_bleu_em = dev_bleu + dev_em

                epoch_stats['dev_bleu'] = dev_bleu
                epoch_stats['dev_em'] = dev_em
                epoch_stats['dev_bleu_em'] = dev_bleu_em
                # Causes circular reference for some reason???
                # epoch_stats['text'] = wandb.Table(data=sampled_texts, columns=['Intent', 'GT', 'Pred'])

                if dev_bleu_em > best_bleu_em:
                    not_bleu_em_inc_cnt = 0
                    logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                    logger.info("  " + "*" * 20)
                    best_bleu_em = dev_bleu_em
                    fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                        cur_epoch, best_bleu_em, dev_bleu, dev_em))
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.data_num == -1 or args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best bleu model into %s", output_model_file)
                else:
                    not_bleu_em_inc_cnt += 1
                    logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                    fa.write(
                        "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(stop_early_str)
                        fa.write(stop_early_str)
                        break
                fa.flush()

            if args.wandb:
                wandb.log(epoch_stats, step=cur_epoch)

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.batch_size)

        criteria = 'best-bleu'
        file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))

        if os.path.isfile(file):
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))

        train_data, valid_data, test_data = load_conala_dataset(args, tokenizer)

        result, sampled_texts = eval_bleu_epoch(args, test_data, test_data.data, model, tokenizer, 'test', criteria)
        test_bleu, test_em = result['bleu'], result['em']
        test_codebleu = result['codebleu'] if 'codebleu' in result else 0
        result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_em, test_codebleu)
        logger.info(result_str)
        fa.write(result_str)
        if args.res_fn:
            with open(args.res_fn, 'a+') as f:
                f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                f.write(result_str)
        if args.wandb:
            wandb.log({
                **result,
                'outputs': wandb.Table(data=sampled_texts, columns=['Intent', 'GT', 'Pred']),
            })
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()

if __name__ == "__main__":
    main()