# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import pickle
import numpy as np

from compute_bleu import compute_bleu

def is_equal(translation_token, tokenized_source):
    if (len(tokenized_source)-1) <= len(translation_token):
        correct_tokens = ((translation_token[:len(tokenized_source) - 1] == tokenized_source[1:]).float()).sum()
        return correct_tokens == (len(tokenized_source) - 1)
    else:
        return False

def compute_metric(translation_corpus, dataset_name, split, tokenizer=None, section=None, args=None,
                   return_data=None):
    dataset = os.path.join('data/{}/{}.json'.format(dataset_name, split))
    with open(dataset) as dataset_file:
        dataset_object = json.loads(dataset_file.read())
    with open('data/{}/{}_order.json'.format(dataset_name, split), 'rb') as f:
        indices = pickle.load(f)
    dataset_object = np.array(dataset_object)[indices].tolist()

    exact_match_acc = 0
    oracle_exact_match_acc = 0
    execution_acc = 0

    if return_data:
        log_data = []

    if section is None:
        section = range(len(dataset_object))
    mistakes = []
    for index in section:
        translation = translation_corpus[index]
        reference = dataset_object[index]['snippet'].lower()
        if return_data and len(log_data) < 100:
            log_data.append([
                reference,
                translation['str'] if isinstance(translation, dict) else translation[0]['str']
            ])
        tokenized_source = tokenizer.encode(reference, padding=True, truncation=True, return_tensors="pt")[0]
        if isinstance(translation, dict):
            if is_equal(translation['token'], tokenized_source.to('cuda')):
                exact_match_acc += 1
            else:
                mistakes.append((index, translation['str']))
            translation = translation['str']
            translation_corpus[index] = translation
        else:
            if is_equal(translation[0]['token'], tokenized_source):
                exact_match_acc += 1
            else:
                mistakes.append((index, translation[0]['str']))
            for trans in translation:
                if is_equal(trans['token'], tokenized_source):
                    oracle_exact_match_acc += 1
                    break
            translation = translation[0]['str']
            translation_corpus[index] = translation

    bleu, bleu_sentence = compute_bleu(translation_corpus, dataset_object, section, args=args)
    metrics = {'bleu': bleu,
               'bleu_sentence': bleu_sentence,
               'exact_match': exact_match_acc/len(section),
               'exact_oracle_match': oracle_exact_match_acc/len(section),
               'exec_acc': execution_acc/len(section),
               'mistakes': mistakes
               }
    if return_data:
        return metrics, log_data
    else:
        return metrics

