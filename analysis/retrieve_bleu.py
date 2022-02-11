from nltk.translate.bleu_score import sentence_bleu
from data.conala.conala import Conala
from transformers import RobertaTokenizer
from evaluator.bleu import (
    compute_pair_bleu_all_prefixes, 
    _make_possible_match_matrix, 
    _make_ratio_matrix,
    _get_suffix_ngrams_list,
    compute_bleu
)
import torch
from tqdm import tqdm
from collections import namedtuple
import pickle

class Args(object):
    mono_min_prob = 0.1
    add_lang_ids = False
    add_task_prefix = False

    def __getattr__(self, name):
        return self.__dict__.get(name)

args = Args()

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

mined_data = Conala('conala', 'train', tokenizer, args, monolingual=True)
doc_data = Conala('conala', 'doc', tokenizer, args, monolingual=False)

test_data = Conala('conala', 'test', tokenizer, args)

# should probably decanonicalize code to avoid artificially low bleu 
# due to identifiers
full_data = mined_data
full_data.data.extend(doc_data.data)

Neighbor = namedtuple('Neighbor', ['data_idx', 'prefix', 'bleu'])

def find_bleu_intent_neighbors(source, dataset, pbar=None):
    iterable = range(len(dataset))
    if pbar is None:
        iterable = tqdm(iterable)

    neighbors = []
    best_bleu = 0
    for i in iterable:
        reference = dataset[i]['intent']['input_ids'][1:-1]
        bleu = compute_bleu([[reference]], [source])[0]
        if bleu > best_bleu:
            best_bleu = bleu
        if bleu > 0.5:
            neighbors.append(i)
        pbar.update(1)
    print(best_bleu)
    1/0
    return neighbors


def find_bleu_neighbors(snippet, dataset, max_dataset_length, pbar=None):
    neighbors = [[] for _ in range(len(snippet))]

    possible_match_matrix = _make_possible_match_matrix(len(snippet), 4)
    ratio_matrix = _make_ratio_matrix(max_dataset_length, len(snippet))

    iterable = range(len(dataset))
    if pbar is None:
        iterable = tqdm(iterable)

    trans_suffix_ngrams = [_get_suffix_ngrams_list(snippet, trans_len, max_order=4) 
                           for trans_len in range(1, len(snippet) + 1)]

    for i in iterable:
        reference = dataset[i]['snippet']['input_ids'][1:-1]
        bleu_matrix = compute_pair_bleu_all_prefixes(reference, snippet,
            possible_match_matrix=possible_match_matrix, 
            ratio_matrix=ratio_matrix[:len(reference)],
            trans_suffix_ngrams=trans_suffix_ngrams)
        bleu_matrix = torch.from_numpy(bleu_matrix)
        # bleu_matrix: |reference| x |snippet|
        best = torch.max(bleu_matrix, dim=0)
        best_neighbor = best.indices
        best_bleu = best.values
        # import pdb; pdb.set_trace()
        for neigh_idx, (prefix, bleu) in enumerate(zip(best_neighbor, best_bleu)):
            if bleu > 0.5:
                neighbors[neigh_idx].append(Neighbor(i, prefix.item(), bleu.item()))
        if pbar is not None:
            pbar.update(1)
    return neighbors

max_dataset_length = max(len(d['snippet']['input_ids']) - 2 for d in full_data.data)

# n = find_bleu_neighbors(test_data[455]['snippet']['input_ids'][1:-1], full_data, max_dataset_length, pbar=None)

with tqdm(total=len(full_data) * len(test_data)) as pbar:
    for i in range(len(test_data)):
        pbar.set_description(f"On test #{i}")
        n = find_bleu_intent_neighbors(test_data[i]['intent']['input_ids'][1:-1], full_data, pbar=pbar)
        with open(f'nn_data/bleu_source_retrievals/test_{i}.csv', 'w+') as f:
            f.write(','.join(str(x) for x in n))

# with tqdm(total=len(full_data) * len(test_data)) as pbar:
#     for i in range(len(test_data)):
#         pbar.set_description(f"On test #{i}")
#         n = find_bleu_neighbors(test_data[i]['snippet']['input_ids'][1:-1], full_data, max_dataset_length, pbar=pbar)

#         with open(f'nn_data/bleu_retrievals/test_{i}', 'wb+') as f:
#             pickle.dump(n, f)