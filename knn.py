'''
Taken from https://github.com/urvashik/knnmt
'''

import os
import pickle
import torch
import torch.nn.functional as F
import warnings
try:
    import faiss
except:
    warnings.warn('Unable to import faiss.')
import numpy as np
import time
try:
    from torch_scatter import scatter
except:
    def scatter(src: torch.Tensor, out: torch.Tensor, 
                index: torch.LongTensor, dim: int):
        out.scatter_(dim, index, src)

def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)

class KNN_Dstore(object):
    def __init__(self, args, vocab_size, pad_idx):
        self.half = args.fp16
        if hasattr(args, "decoder_embed_dim"):
            self.dimension = args.decoder_embed_dim
        else:
            self.dimension = args.knn_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.use_faiss_only = args.use_faiss_only
        self.index = self.setup_faiss(args)
        self.lmbda = args.lmbda
        self.knn_temp = args.knn_temp

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        kv_pairs_path = args.dstore_filename + '_kv_pairs.p'
        if os.path.isfile(kv_pairs_path):
            with open(kv_pairs_path, 'rb') as f:
                self.kv_pairs = pickle.load(f)

    def setup_faiss(self, args):
        if not args.indexfile:
            raise ValueError('Cannot use knnlm without an index.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        if args.knn_q2gpu:
            print("Moving quantizer to GPU")
            index_ivf = faiss.extract_index_ivf(index)
            quantizer = index_ivf.quantizer
            quantizer_gpu = faiss.index_cpu_to_all_gpus(quantizer, ngpu=1)
            index_ivf.quantizer = quantizer_gpu

        index.nprobe = args.probe

        if self.use_faiss_only:
            return index

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int32')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_size, 1)) # use 32 bit values
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int64, mode='r', shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))
        return index

    def get_knns(self, queries):
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def dist_func(self, d, k, q, function=None):
        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.metric_type == 'l2':
                start = time.time()
                knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                return -1 * l2
            return d

        if function == 'dot':
            qsize = q.shape
            return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        if function == 'sqrt':
            return -1 * torch.sqrt(d)

        if function == 'do_not_recomp_l2':
            return -1 * d

        raise ValueError("Invalid knn similarity function!")

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx])

        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        # dists = self.dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)
        probs = log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

    def retrieve(self, queries):
        # queries are [batch, seq_len, hidden]
        batch, seq_len = queries.shape[:2]
        dists, knns = self.get_knns(queries.contiguous().view(-1, queries.size(-1)))  # [Batch * seq len, K]

        nn_vals = self.vals[knns].to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
        nn_vals = nn_vals.view(batch, seq_len, -1)  # [B, S, K]
        nn_keys = self.keys[knns].to(queries.device) # [B, S, K, H]

        dists = dists.view(batch, seq_len, -1)  # [B, S, K]
        knns = knns.view(batch, seq_len, -1)  # [B, S, K]
        return dists, knns, nn_vals, nn_keys

    def get_knn_scores_per_step(self, queries, use_dtype=torch.float32, save_knns=False):#, knn_temp=1.0):
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        dists, knns = self.get_knns(queries)
        # (B x beam_size) x K
        dists = torch.from_numpy(dists).type(dtype=use_dtype).cuda()
        dists = -1 * dists / self.knn_temp # negative dists

        probs = log_softmax(dists, dim=-1).type(dtype=use_dtype)

        # (Bxbeam_size)xK
        if self.use_faiss_only:
            indices = torch.from_numpy(knns).long().cuda()
        else:
            indices = torch.from_numpy(self.vals[knns]).long().cuda()
        indices = indices.view(queries.shape[0], self.k)

        ## TRYING SOMETHING OUT
        unique_indices, mapping = torch.unique(indices, return_inverse=True)
        # (Bxbeam)xKxn where n = num unique vals in indices
        knn_scores_by_index = torch.ones([indices.shape[0], indices.shape[1], len(unique_indices)], dtype=use_dtype).cuda()
        knn_scores_by_index[:] = -10000 #-math.inf
        knn_vals_by_index = torch.ones([indices.shape[0], indices.shape[1], len(unique_indices)]).long().cuda()
        knn_vals_by_index[:] = self.pad_idx

        # (Bxbeam)x1xK
        indices = indices.unsqueeze(2)
        probs = probs.unsqueeze(2)
        mapping = mapping.unsqueeze(2)
        knn_scores_by_index.scatter_(dim=2, index=mapping, src=probs)
        knn_vals_by_index.scatter_(dim=2, index=mapping, src=indices)
        # (Bxbeam)xn
        knn_scores_by_index = knn_scores_by_index.logsumexp(dim=1)
        knn_vals_by_index = knn_vals_by_index.max(dim=1)[0]
        full_knn_scores = torch.ones([queries.shape[0], self.vocab_size], dtype=use_dtype).cuda()
        full_knn_scores[:] = -10000 #-math.inf
        full_knn_scores.scatter_(dim=1, index=knn_vals_by_index, src=knn_scores_by_index)
        ## TRYING SOMETHING OUT

        if save_knns:
            return full_knn_scores, (torch.from_numpy(knns).long().cuda(), probs.squeeze(-1), indices.squeeze(-1))

        return full_knn_scores

    def calculate_select_knn_prob(self,
        distance: torch.Tensor,              # [B, S, K]
        tgt_index: torch.Tensor,             # [B, S, K]
        knn_select_prob: torch.Tensor = None # [B, S, K]
    ):
        '''
        Taken with slight modification from Adaptive kNN-MT
        https://github.com/zhengxxn/adaptive-knn-mt/blob/main/fairseq/modules/knn_datastore.py
        '''
        scaled_dists = -distance / self.knn_temp

        knn_weight = torch.softmax(scaled_dists, dim=-1)  # [B, S, K]
        weight_sum_knn_weight = knn_weight * knn_select_prob

        return self.scatter_knn_scores(weight_sum_knn_weight, tgt_index)

    def scatter_knn_scores(self, knn_scores, tgt_index):
        B, S, K = knn_scores.shape
        knn_tgt_prob = torch.zeros(B, S, K, self.vocab_size).to(knn_scores)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze_(-1)  # [B, S, K, 1]

        scatter(src=knn_scores.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]
        return prob
