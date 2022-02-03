import torch
from data.conala.conala import Conala
from transformers import RobertaTokenizer

class Args(object):
    def __init__(self, **args):
        self.args = args

    def __getattr__(self, name):
        return self.args.get(name)

args = Args(add_lang_ids=False, add_task_prefix=False, batch_size=32, compute_doc=False, compute_mined=False, compute_train=True, dstore_filename='datastore/doc-mined', dstore_fp16=True, dstore_size=1829724, faiss_gpu=True, faiss_metric_type='l2', fp16=False, indexfile='datastore/doc-mined_knn.index', k=32, knn_attn=False, knn_embed_dim=768, knn_q2gpu=False, knn_sim_func=None, knn_temp=1.0, lmbda=0.05, mono_min_prob=0.1, move_dstore_to_mem=True, no_load_keys=True, pretrained_path='pretrained_weights/conala_codet5_base.bin', probe=8, seed=1234, use_faiss_only=False)

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

train_data = Conala('conala', 'train', tokenizer, args, monolingual=False)
val_data = Conala('conala', 'dev', tokenizer, args, monolingual=False)

train_data.data.extend(val_data.data)

vals = []
for data in train_data:
    vals.append(data['snippet']['input_ids'][1:])

nn_data = torch.load('nn_data/train.bin')

token_corr = dict()
token_err = dict()
for val, d in zip(vals, nn_data):
    for gt_token, top1 in zip(val, d[:, 0, 2]):
        if int(gt_token) == int(top1):
            token_corr[gt_token] = token_corr.get(gt_token, 0) + 1
        else:
            token_err[gt_token] = token_err.get(gt_token, 0) + 1

for k in (token_corr.keys() | token_err.keys()):
    token_acc = token_corr.get(k) / (token_corr.get(k) + token_err.get(k))

num_corr = sum(token_corr.values())
num_err = sum(token_err.values())
acc = num_corr / (num_corr + num_err)

print('Top-1 accuracy:', acc)

ranked = sorted(token_acc.items(), key=lambda x: x[1])

print('Easiest tokens:')
for i in range(1, 51):
    print(i, tokenizer.decode(ranked[-i][0]), ranked[-i][1],
        f"{token_corr.get(ranked[-i][0], 0)} / {token_err.get(ranked[-i][0], 0)}")

print('Hardest tokens:')
for i in range(0, 50):
    print(i+1, repr(tokenizer.decode(ranked[i][0])), ranked[i][1], 
        f"{token_corr.get(ranked[i][0], 0)} / {token_err.get(ranked[i][0], 0)}")