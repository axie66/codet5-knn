import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple

from knn import KNN_Dstore
from _modeling_t5 import T5ForConditionalGeneration, ModelOutput


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

    def combine_probs(self, lprobs, knn_scores, lmbda=None):
        '''
        inspired by (but modified from) kNN-MT
        https://github.com/urvashik/knnmt/blob/master/fairseq/sequence_generator.py
        '''
        # lprobs: (batch x beams, vocab_size)
        # knn_scores: (batch x beams, vocab_size)
        assert isinstance(lmbda, (int, list))

        combined = torch.stack([lprobs, knn_scores.to(lprobs)], dim=-1)
        if lmbda is None:
            lmbda = self.lmbda
        if isinstance(lmbda, int):
            lmbda = torch.tensor([1 - lmbda, lmbda])
        else:
            lmbda = torch.tensor(lmbda)

        coeffs = torch.log(lmbda.to(lprobs)).expand_as(combined)
        combined = torch.logsumexp(combined + coeffs, dim=-1)
        return combined

    def forward(self, *args, **kwargs):
        assert not self.training() # kNN only at inference time

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

        knn_scores = self.dstore.get_knn_scores_per_step(query)
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


class T5GatedKNN(T5KNN):
    def __init__(self, *args, k=64, k_embed_dim=0, gate_drop=0.0, use_dists=True, **kwargs):
        super(T5KNN, self).__init__(*args, **kwargs)
        
        self.use_dists = use_dists
        self.learned_k_embed = k_embed_dim > 0
        if self.learned_k_embed:
            self.k_embeddings = nn.Embedding(k, k_embed_dim)
        
        gate_dim = 2 * self.config.hidden_size + k_embed_dim
        self.knn_gate = nn.Sequential(
            nn.Linear(gate_dim, gate_dim),
            nn.Tanh(),
            nn.Dropout(gate_drop),
            nn.Linear(gate_dim, 1)
        )

    def forward(self, *args, **kwargs):
        if not hasattr(self, 'dstore'):
            raise Exception('T5KNN model must be assigned datastore object before being called')
        
        if not (kwargs.get('ret_decoder_ffn_inp') or kwargs.get('ret_encoder_ffn_inp')):
            kwargs['ret_decoder_ffn_inp'] = True

        output = super(T5KNN, self).forward(*args, **kwargs)

        if output.decoder_ffn_inputs:
            queries = output.decoder_ffn_inputs[-1]
        else:
            queries = output.encoder_ffn_inputs[-1]

        distance, knn, nn_vals, nn_keys = self.dstore.retrieve(queries)

        gate_input = [
            queries.unsqueeze(-2).expand(-1, -1, knn.shape[2], -1),
            nn_keys,
        ]
        if self.learned_k_embed:
            batch, seq_len = queries.shape[:2]
            gate_input.append(self.k_embeddings.weight.expand(batch, seq_len, -1, -1))
        gate_input = torch.cat(gate_input, dim=-1) # batch, seq, k, 2 * hidden + k_emb
        
        select_probs = self.knn_gate(gate_input)

        if self.use_dists:
            lprobs = output.logits.log_softmax(dim=-1)
            knn_probs = self.dstore.calculate_select_knn_prob(distance, nn_vals, torch.sigmoid(select_probs))
            knn_lprobs = torch.log_softmax(knn_probs)
            logits = self.combine_probs(lprobs, knn_lprobs, lmbda=[1., 1.])
        else:
            lprobs = knn_lprobs = None
            knn_logits = self.dstore.scatter_knn_scores(select_probs)
            logits = output.logits + knn_logits

        return KNNSeq2SeqLMOutput(
            loss=output.loss,
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
            knn_lprobs=knn_lprobs,
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
