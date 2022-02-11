'''
Pre-compute nearest neighbors to save time during kNN training
'''

import os
import argparse
import random
from tqdm import tqdm
from config import add_knn_args, set_seed

import torch
from knn import KNN_Dstore
from model import T5ForConditionalGeneration
from transformers import RobertaTokenizer

from torch.utils.data import DataLoader
from data.conala.conala import Conala
from dataset import preprocess_batch_conala


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--fp16', action='store_true')

parser.add_argument('--mono_min_prob', type=float, default=0.1)
parser.add_argument('--add_lang_ids', action='store_true')
parser.add_argument('--add_task_prefix', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--pretrained_path', type=str, required=True)
parser.add_argument('--compute-train', action='store_true')
parser.add_argument('--compute-doc', action='store_true')
parser.add_argument('--compute-mined', action='store_true')
parser.add_argument('--compute-test', action='store_true')

parser.add_argument('--seed', type=int, default=None)

parser = add_knn_args(parser)

args = parser.parse_args()
args.faiss_gpu = cuda

print(args)

if args.seed is None:
    args.seed = random.randint(0, 2**15-1)

set_seed(args)

model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

try:
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained_path))
    model.eval()
except:
    print('Unable to load pretrained weights')

dstore = KNN_Dstore(args, vocab_size=tokenizer.vocab_size, pad_idx=tokenizer.pad_token_id)

datasets = []

dataset_str = []
if args.compute_train:
    dataset_str.append('train')
    datasets.append(Conala('conala', 'train', tokenizer, args, monolingual=False))
    datasets.append(Conala('conala', 'dev', tokenizer, args, monolingual=False))
if args.compute_doc:
    dataset_str.append('doc')
    datasets.append(Conala('conala', 'doc', tokenizer, args, monolingual=False))
if args.compute_mined:
    dataset_str.append('mined')
    datasets.append(Conala('conala', 'train', tokenizer, args, monolingual=True))
if args.compute_test:
    dataset_str.append('test')
    datasets.append(Conala('conala', 'test', tokenizer, args, monolingual=False))

dataset_str = '-'.join(dataset_str)

full_dataset = datasets[0]
for i in range(1, len(datasets)):
    full_dataset.data.extend(datasets[i].data)

loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=False, collate_fn=preprocess_batch_conala)

lines = []

nn_data = [] # tuples of distance, neighbor indices, and values
vals = []

with torch.no_grad():
    for i, data in enumerate(tqdm(loader)):
        source_ids = data['source']['input_ids'].to(device)
        source_mask = data['source']['attention_mask'].to(device)
        target_ids = data['target']['input_ids'].to(device)
        target_mask = data['target']['attention_mask'].to(device)
        out = model(input_ids=source_ids, attention_mask=source_mask,
            labels=target_ids, decoder_attention_mask=target_mask,
            ret_decoder_ffn_inp=True)
        queries = out.decoder_ffn_inputs[-1]

        lengths = target_mask.sum(dim=1).cpu()

        dists, knns, nn_vals = dstore.retrieve(queries, ret_keys=False)

        # data[i]: [seq, k, 3], where dim 3 is (distance, neighbor indices, values)
        data = torch.stack((dists, knns, nn_vals), dim=-1)

        for ex, length, val in zip(data, lengths, target_ids):
            nn_data.append(ex[:length])
            vals.append(val[1:1+length])

print('Num tokens:', sum(d.shape[0] for d in nn_data))

if not os.path.isdir('nn_data'):
    os.mkdir('nn_data')
torch.save(nn_data, os.path.join('nn_data', f"{dataset_str}.bin"))
