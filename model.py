import torch

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

    def get_knn_scores(self, query, save_knns=False):
        vocab_size = self.encoder.config.vocab_size
        pad_id = self.tokenizer.pad_token_id
        return self.dstore.get_knn_scores_per_step(
            query, vocab_size, pad_id, save_knns=save_knns
        )

    def interpolate(self, lprobs, knn_scores, lmbda=None):
        '''
        inspired by (but modified from) knnmt/fairseq/sequence_generator.py
        https://github.com/urvashik/knnmt/blob/master/fairseq/sequence_generator.py
        '''
        # lprobs: (batch x beams, vocab_size)
        # knn_scores: (batch x beams, vocab_size)
        combined = torch.stack([lprobs, knn_scores.to(lprobs)], dim=-1)
        if lmbda is None:
            lmbda = self.lmbda
        coeffs = torch.log(torch.tensor([1 - lmbda, lmbda]).to(lprobs)).expand_as(combined)
        combined = torch.logsumexp(combined + coeffs, dim=-1)
        return combined

    def forward(self, *args, lmbda=None, **kwargs):
        if not hasattr(self, 'dstore'):
            raise Exception('T5KNN model must be assigned datastore object before being called')
        
        if not (kwargs.get('decoder_ffn_inputs') or kwargs.get('encoder_ffn_inputs')):
            kwargs['decoder_ffn_inputs'] = True

        output = super(T5KNN, self).forward(*args, **kwargs)

        if lmbda is None:
            lmbda = self.lmbda

        if lmbda == 0.0:
            logits = output.logits
            lprobs = None
            knn_scores = None
        else:
            lprobs = output.logits.log_softmax(dim=-1)
            if output.decoder_ffn_inputs:
                query = output.decoder_ffn_inputs[-1][:, -1]
            else:
                query = output.encoder_ffn_inputs[-1][:, -1]
            knn_scores = self.get_knn_scores(query)
            logits = self.interpolate(lprobs, knn_scores)

        return KNNSeq2SeqLMOutput(
            loss=output.loss,
            logits=logits,
            past_key_values=output.past_key_values,
            decoder_hidden_states=output.hidden_states,
            decoder_attentions=output.attentions,
            decoder_ffn_inputs=output.ffn_inputs,
            cross_attentions=output.cross_attentions,
            encoder_last_hidden_state=output.last_hidden_state,
            encoder_hidden_states=output.hidden_states,
            encoder_attentions=output.attentions,
            encoder_ffn_inputs=output.ffn_inputs,
            model_lprobs=lprobs,
            knn_lprobs=knn_scores,
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
