import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple

from knn import KNN_Dstore
from _modeling_t5 import T5ForConditionalGeneration, ModelOutput

import torch.nn.functional as F
import math

@dataclass
class KNNSeq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_ffn_inputs: Optional[Tuple[torch.FloatTensor]] = None

    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_ffn_inputs: Optional[Tuple[torch.FloatTensor]] = None

    model_lprobs: Optional[torch.FloatTensor] = None
    knn_lprobs: Optional[torch.FloatTensor] = None


class T5KNN(T5ForConditionalGeneration):
    def set_knn_dstore(self, dstore: KNN_Dstore):
        self.dstore = dstore
        self.lmbda = dstore.lmbda

        self.counts = {
            'train': 0,
            'doc': 0,
            'mined': 0
        }

    def combine_probs(self, lprobs, knn_scores, lmbda=None):
        '''
        inspired by (but modified from) kNN-MT
        https://github.com/urvashik/knnmt/blob/master/fairseq/sequence_generator.py
        '''
        # lprobs: (batch x beams, vocab_size)
        # knn_scores: (batch x beams, vocab_size)
        assert lmbda is None or isinstance(lmbda, (float, list))

        combined = torch.stack([lprobs, knn_scores.to(lprobs)], dim=-1)
        if lmbda is None:
            lmbda = float(self.lmbda)
        if isinstance(lmbda, float):
            lmbda = torch.tensor([1 - lmbda, lmbda])
        else:
            lmbda = torch.tensor(lmbda)

        coeffs = torch.log(lmbda.to(lprobs)).expand_as(combined)
        combined = torch.logsumexp(combined + coeffs, dim=-1)
        return combined

    def forward(self, *args, **kwargs):
        assert not self.training # kNN only at inference time

        if not hasattr(self, 'dstore'):
            raise Exception('T5KNN model must be assigned datastore object before being called')
        
        if not (kwargs.get('ret_decoder_ffn_inp') or kwargs.get('ret_encoder_ffn_inp')):
            kwargs['ret_decoder_ffn_inp'] = True

        output = super(T5KNN, self).forward(*args, **kwargs)

        lprobs = output.logits[:, -1].log_softmax(dim=-1)
        if output.decoder_ffn_inputs:
            query = output.decoder_ffn_inputs[-1][:, -1]
        else:
            query = output.encoder_ffn_inputs[-1][:, -1]

        knn_scores, knns = self.dstore.get_knn_scores_per_step(query, ret_knns=True)
        
        # train_knns = knns < 39851
        # doc_knns = knns < 195322
        # knn_types = train_knns + doc_knns
        # self.counts['train'] += int((knn_types == 2).sum().cpu())
        # self.counts['doc'] += int((knn_types == 1).sum().cpu())
        # self.counts['mined'] += int((knn_types == 0).sum().cpu())

        combined_logits = self.combine_probs(lprobs, knn_scores)
        output.logits[:, -1] = combined_logits

        return KNNSeq2SeqLMOutput(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,

            decoder_hidden_states=output.decoder_hidden_states,
            decoder_attentions=output.decoder_attentions,
            decoder_ffn_inputs=output.decoder_ffn_inputs,
            
            cross_attentions=output.cross_attentions,
            
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=output.encoder_hidden_states,
            encoder_attentions=output.encoder_attentions,
            encoder_ffn_inputs=output.encoder_ffn_inputs,
            
            model_lprobs=lprobs,
            knn_lprobs=knn_scores,
        )


class KNNAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=1, dropout_prob=0.0, 
                key_proj=False, query_proj=False):
        super(KNNAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size) if query_proj else None
        self.key = nn.Linear(hidden_size, self.all_head_size) if key_proj else None
        # this dropout is applied after calculating the attention score 
        # following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields 
        # better performance
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else None

    def transform(self, x, linear_layer=None):
        # the corresponding linear_layer of k, v, q are used to project the 
        # hidden_state (x)
        bs, seq_len = x.shape[:2]
        if linear_layer is not None:
            proj = linear_layer(x)
        else:
            proj = x
        # next, we need to produce multiple heads for the proj 
        # this is done by spliting the hidden state to self.num_attention_heads, 
        # each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # by proper transpose, we have proj of 
        # [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        d_k = query.shape[-1]
        # energy: (bs, heads, seq_len, k)
        energy = query @ key.transpose(2, 3) / math.sqrt(d_k)
        energy = torch.where(attention_mask == 0, energy, attention_mask)

        # normalize the scores
        scores = F.softmax(energy, dim=-1)
        scores = self.dropout(scores)

        # attn_value: (bs, heads, seq_len, head_size)
        attn_value = scores @ value

        # [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
        bs, _, seq_len, _ = attn_value.shape
        attn_value = attn_value.transpose(1, 2).reshape(bs, seq_len, -1)
        return attn_value

    def forward(self, model_hidden_states, k_hidden_states, k_embeddings, attention_mask):
        """
        Args:
            `hidden_states`: [bs, seq_len, hidden_dim]
            `k_embeddings`: [bs, seq_len, hidden_dim]
            attention_mask: [bs, seq_len]
        Return:
            `attn_value`: [bs, seq_len, hidden_dim]
        """
        key_layer = self.transform(k_hidden_states, self.key)
        value_layer = self.transform(k_embeddings)
        query_layer = self.transform(model_hidden_states, self.query)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class T5AttnKNN(T5KNN):
    '''
    Attention on nearest neighbors inspired by https://arxiv.org/pdf/2102.02557.pdf
    '''
    def __init__(self, *args, k=64, hidden_size=768, attn_heads=1, attn_dropout_prob=0.0, 
                 key_proj=False, query_proj=False, pos_embeds=False, use_knn_keys=False):
        super(T5AttnKNN, self).__init__()

        self.k = k
        self.use_knn_keys = use_knn_keys

        self.knn_attn = KNNAttention(
            hidden_size=hidden_size, num_attention_heads=attn_heads, 
            dropout_prob=attn_dropout_prob, key_proj=key_proj, query_proj=query_proj
        )

        self.pos_embed = nn.Embedding(k, hidden_size) if pos_embeds else None

        self.interp_linear = nn.Linear(hidden_size, 1)

    def forward(self, *args, **kwargs):
        if not hasattr(self, 'dstore'):
            raise Exception('T5KNN model must be assigned datastore object before being called')
        
        if not (kwargs.get('ret_decoder_ffn_inp') or kwargs.get('ret_encoder_ffn_inp')):
            kwargs['ret_decoder_ffn_inp'] = True

        output = super(T5KNN, self).forward(*args, **kwargs)

        lprobs = output.logits[:, -1].log_softmax(dim=-1)
        if output.decoder_ffn_inputs:
            queries = output.decoder_ffn_inputs[-1][:, -1]
        else:
            queries = output.encoder_ffn_inputs[-1][:, -1]
        
        # interp_coeff: [batch, seq_len, 1]
        interp_coeff = F.sigmoid(self.interp_linear(queries))

        _, _, nn_vals, *nn_keys = self.dstore.retrieve(queries, ret_keys=self.use_knn_keys)
        k_embeddings = self.decoder.embed_tokens(nn_vals)

        if self.use_knn_keys:
            k_hidden_states = nn_keys[0]
        else:
            k_hidden_states = k_embeddings

        if self.pos_embed is not None:
            k_indices = torch.arange(self.k).unsqueeze(0).unsqueeze(0).expand_as(nn_vals)
            k_pos_embeds = self.pos_embed(k_indices)
            k_hidden_states += k_pos_embeds

        # nn_vals: [batch, seq_len, k]
        # nn_keys: [batch, seq_len, k, hidden_dim]
        attn_output = self.knn_attn(
            model_hidden_states=queries, 
            k_hidden_states=k_hidden_states, 
            k_embeddings=k_embeddings
        )

        logits = interp_coeff * attn_output + (1 - interp_coeff) * output.logits
        if 'labels' in kwargs:
            labels = kwargs['labels']
            loss = F.cross_entropy(logits, labels)
        else:
            loss = output.logits

        return KNNSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,

            decoder_hidden_states=output.decoder_hidden_states,
            decoder_attentions=output.decoder_attentions,
            decoder_ffn_inputs=output.decoder_ffn_inputs,
            
            cross_attentions=output.cross_attentions,
            
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=output.encoder_hidden_states,
            encoder_attentions=output.encoder_attentions,
            encoder_ffn_inputs=output.encoder_ffn_inputs,
            
            model_lprobs=lprobs,
            knn_lprobs=attn_output,
        )

if __name__ == '__main__':
    torch.manual_seed(768)
    model = T5KNN.from_pretrained('Salesforce/codet5-base')
    dstore = lambda: ()
    dstore.lmbda = 0.4
    model.set_knn_dstore(dstore)

    lprobs = torch.rand(2, 2, 10)
    knn_scores = torch.rand(2, 2, 10)

    # lprobs = torch.zeros(2, 1, 8) - 1e4
    # knn_scores = torch.zeros(2, 1, 8) - 1e4
    # lprobs[0, 0, 1] = 0.99
    # knn_scores[0, 0, 4] = 0.99

    combined = model.interpolate(lprobs, knn_scores)

    probs = lprobs.softmax(dim=-1)
    knn_probs = knn_scores.softmax(dim=-1)
    expected_combined_probs = probs * 0.6 + knn_probs * 0.4 # numerically unstable

    combined_probs = combined.softmax(dim=-1)

    print(expected_combined_probs)
    print(combined_probs)
