'''
TODO: Check if I'm being an idiot
    - are T5 decoder outputs shifted by 1?

Stores each context (i.e. intent/previous tokens) as a fixed-size embedding

Run with:
python3 store_embeds.py \
    --dataset_name conala \
    --pretrained_path {path/to/pretrained/model}
    --data_type ['train', 'mined', 'csn']

Afterward, run build_dstore.py with appropriate args to generate datastore.
'''

import os
from tqdm import tqdm
import pickle
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

import numpy as np
from data.conala.conala import Conala
from dataset import (
    preprocess_batch_conala, 
    preprocess_batch_concode, 
    load_and_cache_concode_data
)
from model import T5ForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_path', type=str, help='Location of pretrained model')

parser.add_argument('--dataset', type=str, choices=['conala', 'concode'], required=True)

# What kind of data to use in datastore
parser.add_argument('--use_train', action='store_true')
parser.add_argument('--use_doc', action='store_true')
parser.add_argument('--use_mined', action='store_true')

parser.add_argument("--cache_path", type=str)
parser.add_argument("--data_dir", type=str)

parser.add_argument('--save_kv_pairs', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mono_min_prob', type=float, default=0.1)
parser.add_argument('--add_lang_ids', action='store_true')
parser.add_argument('--add_task_prefix', action='store_true')
args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = Model('bert-base-uncased', args)
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

try:
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained_path))
    model.eval()
except:
    print('Unable to load pretrained weights')

dataset_str = []

datasets = []

if args.dataset == 'conala':
    if args.use_train:
        dataset_str.append('train')
        datasets.append(Conala('conala', 'train', tokenizer, args, monolingual=False))
        datasets.append(Conala('conala', 'dev', tokenizer, args, monolingual=False))
    if args.use_doc:
        dataset_str.append('doc')
        datasets.append(Conala('conala', 'doc', tokenizer, args, monolingual=False))
    if args.use_mined:
        dataset_str.append('mined')
        datasets.append(Conala('conala', 'train', tokenizer, args, monolingual=True))

    full_dataset = datasets[0]
    for i in range(1, len(datasets)):
        full_dataset.data.extend(datasets[i].data)
else: # concode -> just train data
    args.data_num = -1
    args.local_rank = -1
    args.max_source_length = 320
    args.calc_stats = False
    examples, full_dataset = load_and_cache_concode_data(args, 'data/concode/dataset/concode/train.json', tokenizer, 'train')

dataset_str = '-'.join(dataset_str)

preprocess_batch = {
    'concode': preprocess_batch_concode,
    'conala': preprocess_batch_conala
}[args.dataset]

loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=False, 
                    collate_fn=preprocess_batch)

dstore_size = 0
# for i in range(len(full_dataset)):
#     example = full_dataset[i]
#     dstore_size += (len(example['snippet']['input_ids']) - 1)

with torch.no_grad():
    for data in tqdm(loader):
        if args.dataset == 'conala':
            mask = data['target']['attention_mask']
        else:
            mask = (data[1] != 0)
        dstore_size += mask.sum()

key_size = model.encoder.config.hidden_size # for no encoder context
# key_size = model.encoder.config.hidden_size * 2 # for encoder context

print('Total # of target tokens:', dstore_size)
print('Size of each key:', key_size)

if not os.path.isdir('datastore'):
    os.mkdir('datastore')

dstore_keys = np.memmap(f'datastore/{args.dataset}_{dataset_str}_keys.npy', dtype=np.float16, mode='w+',
                        shape=(dstore_size, key_size))
dstore_vals = np.memmap(f'datastore/{args.dataset}_{dataset_str}_vals.npy', dtype=np.int32, mode='w+',
                        shape=(dstore_size, 1))

if args.save_kv_pairs:
    kv_pairs = []

with torch.no_grad():
    offset = 0
    for i, data in enumerate(tqdm(loader)):
        if args.dataset == 'conala':
            source_ids = data['source']['input_ids'].to(device)
            source_mask = data['source']['attention_mask'].to(device)
            target_ids = data['target']['input_ids'].to(device)
            target_mask = data['target']['attention_mask'].to(device)
        else:
            source_ids, target_ids = [x.to(device) for x in data]
            source_mask = (source_ids != 0).to(device)
            target_mask = (target_ids != 0).to(device)

        out = model(input_ids=source_ids, attention_mask=source_mask,
            labels=target_ids, decoder_attention_mask=target_mask,
            ret_decoder_ffn_inp=True)
        knn_context = out.decoder_ffn_inputs[-1]
        # target_ids = data['target']['input_ids']
        lengths = target_mask.sum(dim=1).cpu()
        first = i == 0
        for embed, length, ids in zip(knn_context, lengths, target_ids):
            actual_length = length
            if args.save_kv_pairs:
                for i in range(actual_length):
                    if i == 0:
                        context = ''
                    else:
                        context = tokenizer.decode(ids[:i].cpu().tolist())
                    target = tokenizer.decode(int(ids[i])).replace(' ', '')
                    kv_pairs.append((context, target))
            dstore_keys[offset:offset+actual_length] = \
                embed[:actual_length, -key_size:].cpu().numpy().astype(np.float16)
            dstore_vals[offset:offset+actual_length] = \
                ids[:actual_length].view(-1, 1).cpu().numpy().astype(np.int32)
            offset += actual_length
        if first and args.save_kv_pairs:
            print(kv_pairs[:10])

dstore_keys.flush()
dstore_vals.flush()

print('Finished saving vectors.')

if args.save_kv_pairs:
    with open(f'datastore/{dataset_str}_kv_pairs.p', 'wb+') as f:
        pickle.dump(kv_pairs, f)
