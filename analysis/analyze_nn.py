import torch
from data.conala.conala import Conala
from transformers import RobertaTokenizer
from knn import KNN_Dstore
from model import T5ForConditionalGeneration
from dataset import preprocess_batch_conala
import pickle
import sys


class Args(object):
    def __init__(self, **args):
        self.args = args

    def __getattr__(self, name):
        return self.args.get(name)


args = Args(add_lang_ids=False, add_task_prefix=False, batch_size=32, compute_doc=False, compute_mined=False, compute_train=True, dstore_filename='datastore/doc-mined', dstore_fp16=True, dstore_size=1829724, faiss_gpu=True, faiss_metric_type='l2', fp16=False, indexfile='datastore/doc-mined_knn.index',
            k=32, knn_attn=False, knn_embed_dim=768, knn_q2gpu=False, knn_sim_func=None, knn_temp=1.0, lmbda=0.05, mono_min_prob=0.1, move_dstore_to_mem=True, no_load_keys=True, pretrained_path='pretrained_weights/conala_codet5_base.bin', probe=8, seed=1234, use_faiss_only=False)

args.dstore_size = 1948358
# args.k = 16
# args.temp = 100
# args.lmbda = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
model.load_state_dict(torch.load(args.pretrained_path))
model.to(device)

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

dstore = KNN_Dstore(args, vocab_size=tokenizer.vocab_size,
                    pad_idx=tokenizer.pad_token_id)

# train_data = Conala('conala', 'train', tokenizer, args, monolingual=False)
# val_data = Conala('conala', 'dev', tokenizer, args, monolingual=False)

# train_data.data.extend(val_data.data)

data = Conala('conala', 'test', tokenizer, args, monolingual=False)

loader = torch.utils.data.DataLoader(data, batch_size=1,
                                     collate_fn=preprocess_batch_conala,
                                     shuffle=False)

# with open(args.dstore_filename + '_kv_pairs.p', 'rb') as f:
#     kv_pairs = pickle.load(f)

# vals = []
# for data in train_data:
#     vals.append(data['snippet']['input_ids'])

# nn_data = torch.load('nn_data/train.bin')

# token_corr = dict()
# token_err = dict()

output_file = open('nn_data/test_retrievals.txt', 'w+')
sys.stdout = output_file

def _unicode_safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        pass

def print_knn_results(source_ids, target_ids, model_probs, knn_probs, idx=-1):
    '''
    Prints knn results for single example (not batch)

    Args:
        `source_ids`: [src_seq_len]
        `target_ids`: [tgt_seq_len]
        `model_probs`: [tgt_seq_len, vocab_size]
        `knn_probs`: [tgt_seq_len, vocab_size]
    '''
    knn_improved = set()
    knn_correct = set()
    model_wrong = set()

    source = tokenizer.decode(source_ids)
    target = tokenizer.decode(target_ids)
    _unicode_safe_print('Source:', source)
    _unicode_safe_print('Target:', target)
    for seq_idx in range(target_ids.shape[-1]):
        prev_gen = tokenizer.decode(target_ids[:seq_idx] if seq_idx > 0 else 0)
        _unicode_safe_print('Previous:', repr(prev_gen))
        # contexts = [kv_pairs[idx] for idx in knns[seq_idx]]

        a = torch.topk(model_probs[seq_idx], k=5)  # (k,)
        model_topk_probs, model_topk_indices = a.values, a.indices
        b = torch.topk(knn_probs[seq_idx], k=5)  # (k,)
        knn_topk_probs, knn_topk_indices = b.values, b.indices
        print_data = []

        gt_idx = target_ids[seq_idx]
        status_str = []
        if knn_probs[seq_idx, gt_idx] > model_probs[seq_idx, gt_idx]:
            status_str.append('kNN improves over model')
            knn_improved.add((idx, seq_idx))
        if knn_probs[seq_idx].argmax() == gt_idx:
            status_str.append('kNN top pred is correct')
            knn_correct.add((idx, seq_idx))
        if model_probs[seq_idx].argmax() != gt_idx:
            status_str.append('model got it wrong')
            model_wrong.add((idx, seq_idx))

        if status_str:
            print(', '.join(status_str))

        for top_idx in range(5):
            mtoken = repr(tokenizer.decode(model_topk_indices[top_idx]))
            ktoken = repr(tokenizer.decode(knn_topk_indices[top_idx]))
            mprob = float(model_topk_probs[top_idx])
            kprob = float(knn_topk_probs[top_idx])
            print_data.append([mtoken, ktoken, mprob, kprob])
        max_mtoken_length = max(len(x[0]) for x in print_data)
        max_ktoken_length = max(len(x[1]) for x in print_data)

        for i, (mtoken, ktoken, mprob, kprob) in enumerate(print_data):
            _unicode_safe_print(
                f'({i})',
                mtoken.rjust(max_mtoken_length), '=', f'{mprob:.2f}',
                '|',
                ktoken.rjust(max_ktoken_length), '=', f'{kprob:.2f}'
            )
        gt = repr(tokenizer.decode(gt_idx))
        _unicode_safe_print('Ground truth token:', gt)
        print('---')

        # while True:
        #     inp = input('> ')
        #     if inp == 'k' or inp == '':
        #         break
        #     elif inp == 'c':
        #         max_ctx = max(len(c[0]) for c in contexts)
        #         for cidx, (ctx, nxt) in enumerate(contexts):
        #             cidx_str = str(cidx).rjust(2 if args.k > 9 else 1)
        #             ctx = ctx[3:]  # get rid of <s> token
        #             print(f'({cidx_str})', repr(ctx).rjust(max_ctx), '->', nxt)
        # if inp == 'k':
        #     break

    print('\n*********************************\n')

    return knn_improved, knn_correct, model_wrong


knn_improved = set()
knn_correct = set()
model_wrong = set()
total = 0

for i, data in enumerate(loader):
    source_ids = data['source']['input_ids'].to(device)
    source_mask = data['source']['attention_mask'].to(device)
    target_ids = data['target']['input_ids'].to(device)
    target_mask = data['target']['attention_mask'].to(device)
    out = model(input_ids=source_ids, attention_mask=source_mask,
                labels=target_ids, decoder_attention_mask=target_mask,
                ret_decoder_ffn_inp=True)

    queries = out.decoder_ffn_inputs[-1]

    knn_scores, knns = dstore.get_knn_scores_per_step(queries, ret_knns=True)

    model_probs = out.logits.softmax(dim=-1)
    knn_probs = torch.exp(knn_scores)

    model_probs = model_probs.squeeze(0)  # (seq_len, vocab)
    knn_probs = knn_probs.squeeze(0)  # (seq_len, vocab)

    source_ids = source_ids.squeeze(0)
    target_ids = target_ids.squeeze(0)

    a, b, c = print_knn_results(
        source_ids, target_ids, model_probs, knn_probs, idx=i)
    knn_improved.update(a)
    knn_correct.update(b)
    model_wrong.update(c)
    total += model_probs.shape[0]

output_file.close()

print('% knn correct:', len(knn_correct) / total)
print('% knn improved:', len(knn_improved) / total)
print('% knn correct given model incorrect:', len(knn_correct & model_wrong) / len(model_wrong))


# for val, d in zip(vals, nn_data):
#     for gt_token, top1 in zip(val, d[:, 0, 2]):
#         if int(gt_token) == int(top1):
#             token_corr[gt_token] = token_corr.get(gt_token, 0) + 1
#         else:
#             token_err[gt_token] = token_err.get(gt_token, 0) + 1

# token_acc = {}
# for k in (token_corr.keys() | token_err.keys()):
#     token_acc[k] = token_corr.get(k, 0) / (token_corr.get(k, 0) + token_err.get(k, 0))

# num_corr = sum(token_corr.values())
# num_err = sum(token_err.values())
# acc = num_corr / (num_corr + num_err)

# print('Top-1 accuracy:', acc)

# ranked = sorted(token_acc.items(), key=lambda x: x[1])

# print('Easiest tokens:')
# i = 1
# seen = 0
# while seen < 50:
#     if token_corr.get(ranked[-i][0], 0) + token_err.get(ranked[-i][0], 0) < 5:
#         i += 1
#         continue
#     print(i, tokenizer.decode(ranked[-i][0]), ranked[-i][1],
#         f"{token_corr.get(ranked[-i][0], 0)} / {token_err.get(ranked[-i][0], 0)}")
#     i += 1
#     seen += 1

# print('Hardest tokens:')
# i = 0
# seen = 0
# while seen < 50:
#     if token_corr.get(ranked[i][0], 0) + token_err.get(ranked[i][0], 0) < 5:
#         i += 1
#         continue
#     print(i+1, repr(tokenizer.decode(ranked[i][0])), ranked[i][1],
#         f"{token_corr.get(ranked[i][0], 0)} / {token_err.get(ranked[i][0], 0)}")
#     i += 1
#     seen += 1
