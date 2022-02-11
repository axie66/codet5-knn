import pickle
import os
from collections import namedtuple
from tkinter import N
import torch
from data.conala.conala import Conala
from transformers import RobertaTokenizer
import numpy as np
from evaluator.bleu import compute_bleu

class DataArgs(object):
    mono_min_prob = 0.1
    add_lang_ids = False
    add_task_prefix = False

args = DataArgs()

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

test_data = Conala('conala', 'test', tokenizer, args, monolingual=False)

doc_data = Conala('conala', 'doc', tokenizer, args, monolingual=False)
mined_data = Conala('conala', 'train', tokenizer, args, monolingual=True)

ret_dir = 'nn_data/bleu_retrievals'
ret_files = os.listdir(ret_dir)

def get_from_datasets(i, dset1, dset2):
    if i < len(dset1):
        return dset1[i]
    else:
        return dset2[i - len(dset1)]

Neighbor = namedtuple('Neighbor', ['data_idx', 'prefix', 'bleu'])

bleu_retrievals = [None] * 500

for filename in ret_files:
    full_filename = ret_dir + '/' + filename
    idx = int(filename.split('_')[-1])
    with open(full_filename, 'rb') as f:
        bleu_retrievals[idx] = pickle.load(f)

# knn_retrievals[i]: tensor of shape [seq_len, k, 3] 
#   where dim 2 is (distance, neighbor indices, values)
knn_retrievals = torch.load('nn_data/test.bin', map_location='cpu')

neighbor2example = []
for i in range(len(doc_data)):
    num = len(doc_data[i]['snippet']['input_ids'])
    doc_data[i]['id'] = i
    neighbor2example.extend([(i, n) for n in range(num)])

for i in range(len(mined_data)):
    num = len(mined_data[i]['snippet']['input_ids'])
    mined_data[i]['id'] = i + len(doc_data)
    neighbor2example.extend([(i + len(doc_data), n) for n in range(num)])

test_idx = 455
snippet_length = 6

snippet = test_data[test_idx]['snippet']['input_ids'][1:-1][:snippet_length]
print('snippet context:\n\t', tokenizer.decode(snippet))

def get_knn_contexts(knn_retrievals, test_idx, prefix_length, limit=10):
    knn_ret = knn_retrievals[test_idx][prefix_length+1]
    knn_contexts = []
    for _, neighbor_idx, v in knn_ret[:limit]:
        dstore_idx, length = neighbor2example[int(neighbor_idx)]
        *ctx, val = get_from_datasets(dstore_idx, doc_data, mined_data)['snippet']['input_ids'][:length+1]
        assert val == v
        knn_contexts.append(
            (tokenizer.decode(ctx), tokenizer.decode(val))
        )
    return knn_contexts

def get_bleu_contexts(bleu_retrievals, test_idx, prefix_length, limit=10):
    bleu_contexts = []
    rets = bleu_retrievals[test_idx][prefix_length-1]
    rets.sort(key=lambda x: x.bleu, reverse=True)
    for neighbor in rets[:limit]:
        dstore_idx, prefix, bleu = neighbor
        *ctx, val = get_from_datasets(dstore_idx, mined_data, doc_data)['snippet']['input_ids'][1:prefix+3]
        # sanity_bleu, *_ = compute_bleu([[ctx]], [snippet])
        # assert sanity_bleu == bleu # bleu computation was slightly off
        bleu_contexts.append(
            (tokenizer.decode(ctx), tokenizer.decode(val))
        )
    return bleu_contexts

paired_retrievals = []
for bleu_ret, knn_ret in zip(bleu_retrievals, knn_retrievals):
    # bleu_ret: [seq_len', num_neighbors] where seq_len' = seq_len - 2
    # knn_ret: [seq_len, k, 3]
    paired_retrievals.append(zip(bleu_ret, knn_ret[2:]))


def get_bleu_ids(bleu_ret):
    ids = set()
    for neighbor in bleu_ret:
        i = get_from_datasets(neighbor[0], mined_data, doc_data)
        ids.add(i['id'])
    return ids

def get_knn_ids(knn_ret):
    ids = set()
    global i, neighbor_idx
    for _, dstore_idx, _ in knn_ret:
        neighbor_idx, _ = neighbor2example[int(dstore_idx)]
        i = get_from_datasets(neighbor_idx, doc_data, mined_data)
        assert i['id'] == int(neighbor_idx)
        ids.add(i['id'])
    return ids

intersect = [
    [
        (lambda s: (s, (len(s), len(bleu_ret))))(
            get_bleu_ids(bleu_ret) & get_knn_ids(knn_ret)
        )
        for bleu_ret, knn_ret in x
    ] 
    for x in paired_retrievals
]

ratios = []
for a in intersect:
    for _, r in a:
        if r[1] > 10:
            ratios.append(r)

weight_avg = sum(r[0] for r in ratios) / sum(min(r[1], 64) for r in ratios)

avg = sum(r[0] / min(r[1], 64) for r in ratios) / len(ratios)

output_file = open('nn_data/bleu_vs_knn.txt', 'w+')

for i in range(500):
    snippet = test_data[i]['snippet']['input_ids'][1:-1]
    source = test_data[i]['intent']['input_ids'][1:-1]
    output_file.write(f"Example #{i}\n")
    output_file.write(f"Source: {tokenizer.decode(source)}\n")
    output_file.write(f"Snippet: {tokenizer.decode(snippet)}\n")
    for prefix_length in range(len(snippet)):
        output_file.write(f"\t[{prefix_length+1}] Context: {tokenizer.decode(snippet[:prefix_length+1])}\n")
        bleu_contexts = get_bleu_contexts(bleu_retrievals, i, prefix_length+1, limit=10)
        knn_contexts = get_knn_contexts(knn_retrievals, i, prefix_length+1, limit=10)
        if len(bleu_contexts) == 0:
            output_file.write('\tNo BLEU retrievals, skipping...\n')
            continue
        output_file.write('\tBLEU retrievals:\n')
        output_file.write('\n'.join(
            f'\t\t({i}) {repr(ctx)} ==> {repr(val)}' for i, (ctx, val) in enumerate(bleu_contexts)
        ))
        output_file.write('\n')
        output_file.write('\tkNN retrievals:\n')
        output_file.write('\n'.join(
            f'\t\t({i}) {repr(ctx)} ==> {repr(val)}' for i, (ctx, val) in enumerate(knn_contexts)
        ))
        output_file.write('\n\n')

    output_file.write('\n' + '*' * 16 + '\n')


# num_neighbors = [np.array([len(n) for n in x]) for x in bleu_retrievals]

# neigh_counts = np.zeros(max(len(n) for n in num_neighbors))
# for n in num_neighbors:
#     neigh_counts[:len(n)] += n

# import matplotlib.pyplot as plt

# plt.plot(np.arange(1, len(neigh_counts) + 1), neigh_counts, linestyle='--', marker='o', color='b')

# plt.xlabel('Context length')
# plt.ylabel('# of bleu > 0.5')
# plt.show()
