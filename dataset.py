import os
from tqdm import tqdm
import json
import time
import random
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from data.conala.conala import Conala

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        ex = self.features[i]
        return ex.source_ids, ex.target_ids, len(ex.target_ids)

    def __len__(self):
        return len(self.features)

def preprocess_batch_concode(batch):
    xs = []
    ys = []
    # y_lens = []
    for x, y, y_len in batch:
        xs.append(x)
        ys.append(y)
        # y_lens.append(y_len)
    # xs: (batch, src_seq_len)
    # ys: (batch, tgt_seq_len)
    if len(ys[0]) == 0: # source only
        return pad_sequence(xs)
    return pad_sequence(xs, batch_first=True), pad_sequence(ys, batch_first=True)#, torch.tensor(y_lens)

def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(tqdm(f)):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    if args.max_source_length > 0:
        source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    else:
        source_ids = tokenizer.encode(source_str, max_length=512, truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1

    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        target_str = target_str.replace('</s>', '<unk>')
        if args.max_target_length > 0:
            target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                          truncation=True)
        else:
            target_ids = tokenizer.encode(target_str)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        torch.tensor(source_ids),
        torch.tensor(target_ids),
        url=example.url
    )

def load_and_cache_concode_data(args, filename, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_concode_examples(filename, args.data_num)

    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))
    if args.calc_stats:
        if split_tag == 'train':
            calc_stats(examples, tokenizer, is_tokenize=True)
        else:
            calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = [convert_examples_to_features(ex) for ex in tqdm(tuple_examples, total=len(tuple_examples))]
        data = SimpleDataset(features)
        # all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        # if split_tag == 'test' or only_src:
        #     data = TensorDataset(all_source_ids)
        # else:
        #     all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        #     data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data

def calc_stats(examples, tokenizer=None, is_tokenize=False, plot=True):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in tqdm(examples):
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))

    if plot:
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].hist(avg_src_len)
        ax[0, 0].set_title('Source Lengths')

        ax[0, 1].hist(avg_trg_len)
        ax[0, 1].set_title('Target Lengths')

        ax[1, 0].hist(avg_src_len_tokenize)
        ax[1, 0].set_title('Tokenized Source Lengths')

        ax[1, 1].hist(avg_trg_len_tokenize)
        ax[1, 1].set_title('Tokenized Target Lengths')
        plt.show()
    
    return avg_src_len, avg_trg_len, avg_src_len_tokenize, avg_trg_len_tokenize

def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)

def load_conala_dataset(args, tokenizer):
    splits = ['train', 'dev', 'test']
    datasets = []
    for split in splits:
        dataset = Conala('conala', split, tokenizer, args)
        datasets.append(dataset)
    return (*datasets,) if len(datasets) > 1 else dataset

def preprocess_batch_conala(data):
    data_intents = [d['intent'] for d in data]
    data_snippets = [d['snippet'] for d in data]
    keys = ['input_ids', 'attention_mask',] #'token_type_ids']
    source_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_intents], batch_first=True, padding_value=0)
                              for key in keys}
    target_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_snippets], batch_first=True, padding_value=0)
                                for key in keys}
    return {'source': source_dict, 'target': target_dict}
