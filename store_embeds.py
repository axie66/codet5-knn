'''
Stores each context (i.e. intent + previous tokens) as a fixed-size embedding

Afterward, run build_dstore.py with appropriate args to generate datastore.
'''

import os
from tqdm import tqdm
import pickle
import argparse

import torch
from torch.utils.data import DataLoader

import numpy as np

from config import *
from transformers import RobertaTokenizer
from model import T5ForConditionalGeneration

from dataset import load_and_cache_gen_data, collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_path', type=str, help='Location of pretrained model', required=True)
parser.add_argument('--dataset_path', type=str, help='Location of saved pytorch dataset object (as a .pt file) used for datastore', required=True)
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--save_kv_pairs', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
model.to(device)
model.device = device
model.load_state_dict(torch.load(args.pretrained_path))
model.eval()

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

dataset = torch.load(args.dataset_path)

loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

dstore_size = 0
# for i in range(len(full_dataset)):
#     example = full_dataset[i]
#     dstore_size += (len(example['snippet']['input_ids']) - 1)

with torch.no_grad():
    for _, _, lengths in tqdm(loader):
        dstore_size += (lengths.sum() - len(lengths))

key_size = model.decoder.config.hidden_size

print('Total # of target tokens:', dstore_size)
print('Size of each key:', key_size)

if not os.path.isdir('datastore'):
    os.mkdir('datastore')

dstore_keys = np.memmap(f'datastore/{args.data_type}_keys.npy', dtype=np.float16, mode='w+',
                        shape=(dstore_size, key_size))
dstore_vals = np.memmap(f'datastore/{args.data_type}_vals.npy', dtype=np.int32, mode='w+',
                        shape=(dstore_size, 1))

kv_pairs = []

with torch.no_grad():
    offset = 0
    for i, (input_ids, labels, lengths) in enumerate(tqdm(loader)):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        out = model(input_ids=input_ids, labels=labels, ret_decoder_ffn_inp=True)
        knn_context = out.decoder_ffn_inputs
        for embed, length, ids in zip(knn_context, lengths, labels):
            actual_length = length-1
            if args.save_kv_pairs:
                for i in range(actual_length):
                    context = model.tokenizer.decode(ids[:i+1].cpu().tolist())
                    target = model.tokenizer.decode(int(ids[i+1])).replace(' ', '')
                    kv_pairs.append((context, target))
            dstore_keys[offset:offset+actual_length] = \
                embed[:actual_length].cpu().numpy().astype(np.float16)
            dstore_vals[offset:offset+actual_length] = \
                ids[1:1+actual_length].view(-1, 1).cpu().numpy().astype(np.int32)
            offset += actual_length
        if i == 0 and args.save_kv_pairs:
            print(kv_pairs[:10])

dstore_keys.flush()
dstore_vals.flush()

print('Finished saving vectors.')

if args.save_kv_pairs:
    with open(f'datastore/{args.data_type}_kv_pairs.p', 'wb+') as f:
        pickle.dump(kv_pairs, f)
