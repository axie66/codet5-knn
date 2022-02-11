# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
from typing import List, Dict, Tuple
import numpy as np
# import numba


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _get_suffix_ngrams_counter(sentence: List[int], end_idx: int, 
                               max_order=4) -> collections.Counter:
    return collections.Counter(tuple(sentence[start_idx:end_idx]) for start_idx in range(max(0, end_idx - max_order), end_idx))


def _add_suffix_ngrams_dict(ngrams_dict: Dict[Tuple[int, ...], int], 
                            sentence: List[int], end_idx: int, 
                            max_order=4):
    for start_idx in range(max(0, end_idx - max_order), end_idx):
        ngram = tuple(sentence[start_idx:end_idx])
        ngrams_dict[ngram] = ngrams_dict.get(ngram, 0) + 1


def _get_suffix_ngrams_list(sentence: List[int], end_idx: int, 
                            max_order=4) -> List[Tuple[int,...]]:
    return [tuple(sentence[start_idx:end_idx]) for start_idx in range(max(0, end_idx - max_order), end_idx)]


def _make_possible_match_matrix(translation_length: int, max_order: int, smoothed: bool = False) -> np.ndarray:
    possible_match_matrix = np.expand_dims(np.arange(translation_length), 1) - np.expand_dims(np.arange(max_order), 0) + 1
    possible_match_matrix = np.maximum(possible_match_matrix, 0 if smoothed else 1)
    possible_match_matrix = np.expand_dims(possible_match_matrix, 0)
    return possible_match_matrix


def _make_ratio_matrix(reference_length: int, translation_length: int) -> np.ndarray:
    ratio_matrix = np.expand_dims(np.arange(1, translation_length + 1), 0) / np.expand_dims(np.arange(1, reference_length + 1), 1)
    ratio_matrix = np.minimum(ratio_matrix, 1.0)
    return ratio_matrix


def compute_pair_bleu_all_prefixes(reference: List[int], translation: List[int],
                                    possible_match_matrix: np.ndarray = None,
                                    ratio_matrix: np.ndarray = None,
                                    trans_suffix_ngrams: np.ndarray = None,
                                    max_order: int = 4, 
                                    smoothed: bool = False):
    if possible_match_matrix is None or possible_match_matrix.shape != (1, len(translation), max_order):
        # 1/0 
        possible_match_matrix = _make_possible_match_matrix(len(translation), max_order, smoothed=smoothed)

    if trans_suffix_ngrams is None:
        # 1/0
        trans_suffix_ngrams = [_get_suffix_ngrams_list(translation, trans_len, max_order=max_order) 
                               for trans_len in range(1, len(translation) + 1)]

    if ratio_matrix is None or ratio_matrix.shape != (len(reference), len(translation)):
        # 1/0
        ratio_matrix = _make_ratio_matrix(len(reference), len(translation))

    return _compute_pair_bleu_all_prefixes(reference, translation,
        possible_match_matrix, ratio_matrix, trans_suffix_ngrams, 
        max_order=max_order, smoothed=smoothed)

# import numba
# @numba.jit
def _compute_pair_bleu_all_prefixes(reference: List[int], translation: List[int],
                                    possible_match_matrix: np.ndarray,
                                    ratio_matrix: np.ndarray,
                                    trans_suffix_ngrams: np.ndarray,
                                    max_order: int = 4, 
                                    smoothed: bool = False) -> np.ndarray:
    '''
    Args:
        reference: Ground truth sentence
        translation: Translation sentence
        max_order: maximum n-gram order used for BLEU score
    Returns:
        bleu: Matrix of size |reference| x |translation| 
              where bleu[i][j] = bleu(reference[:i+1], translation[:j+1])
              (i.e. (i+1)-length prefix of reference, (j+1)-length prefix 
              of translation)
    '''
    match_matrix = np.zeros((len(reference), len(translation), max_order), dtype=np.int16)

    ref_ngrams = dict()
    for ref_len in range(len(reference)):
        # ref_ngrams += _get_suffix_ngrams_counter(reference, ref_len+1, max_order=max_order)
        _add_suffix_ngrams_dict(ref_ngrams, reference, ref_len+1, max_order=max_order)
        used_ngrams = dict()
        for trans_len in range(len(translation)):
            # count_overlaps = np.zeros((max_order,))
            if trans_len > 0:
                # count_overlaps[:] = match_matrix[ref_len, trans_len-1]
                match_matrix[ref_len, trans_len] = match_matrix[ref_len, trans_len-1]
            for ngram in trans_suffix_ngrams[trans_len]:
                ref_count = ref_ngrams.get(ngram, 0)
                used_count = used_ngrams.get(ngram, 0)
                if used_count < ref_count:
                    # count_overlaps[len(ngram) - 1] += 1
                    used_ngrams[ngram] = used_count + 1
                    match_matrix[ref_len, trans_len, len(ngram)-1] += 1
                elif ref_count and used_count >= ref_count:
                    print(ngram, 'not counted', ref_count, used_count)
            # match_matrix[ref_len, trans_len, :] = count_overlaps

    if smoothed:
        # with smoothing
        smoothed_precisions = (match_matrix + 1) / (possible_match_matrix + 1)

        log_avg = np.sum(np.log(smoothed_precisions), axis=-1) / max_order
        geo_mean = np.exp(log_avg)

    else:
        precisions = match_matrix / possible_match_matrix
        zero_bleu = np.any(precisions == 0, axis=-1)
        # the next line will give warnings 
        # but its ok since we don't pick the funky elements anyway
        log_avg = np.sum(np.log(precisions), axis=-1) / max_order
        geo_mean = np.where(zero_bleu, 0.0, np.exp(log_avg))

    bp = np.exp(1 - 1 / ratio_matrix)

    bleu = geo_mean * bp
    return bleu, precisions[-1, -1]


def _bleu(ref_file, trans_file, subword_option=None):
    max_order = 4
    smooth = True
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            translations.append(line.strip().split())
    bleu_score, _, _, _, _, _ = compute_bleu(
        per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)


if __name__ == '__main__':
    # reference = ['<s>', 'cat', 'in', 'the', 'hat', 'trick']
    # translation = ['<s>', 'cat', 'in', 'the', 'hot']

    # reference = [4288, 18, 1289, 67, 3446, 2668, 17, 88, 2187, 1570, 2218, 2121, 2187, 2809, 2218, 2187, 14440, 2218, 15, 6134]
    # translation = [4288, 18, 1289, 67, 67, 67, 67, 76, 90, 13, 364, 331, 316, 284, 4773, 67, 9724, 18, 2372, 10756]

    reference = [9499, 18, 803, 18, 6387, 12, 20, 16, 11597, 6134]
    translation = [9499, 18, 803, 18, 6387, 12]

    b = compute_pair_bleu_all_prefixes(reference, translation, smoothed=False, max_order=3)

    import time
    start = time.time()
    b, *x = compute_pair_bleu_all_prefixes(reference, translation, smoothed=False, max_order=3)
    print('Fast took', time.time() - start)
    # m = m.astype(np.int32)

    # mm = [[str(tuple(x)) for x in y] for y in m]
    # pp = [[str(tuple(x)) for x in y] for y in p]

    gt_bleu = np.zeros((len(reference), len(translation)))
    start = time.time()
    for ref_len in range(1, len(reference) + 1):
        for trans_len in range(1, len(translation) + 1):
            gt_bleu[ref_len-1][trans_len-1] = compute_bleu([[reference[:ref_len]]], [translation[:trans_len]], smooth=False, max_order=3)[0]
    print('Brute force took', time.time() - start)

    np.set_printoptions(linewidth=1000, formatter={'float': lambda x: "{0:0.4f}".format(x)})
    # print(np.array(mm))
    # print(np.array(pp))
    # print()

    print(b)

    print(gt_bleu)

    print('close to ground truth?', np.allclose(gt_bleu, b))

    print(compute_bleu([[reference[:ref_len]]], [translation[:trans_len]], smooth=False, max_order=3))
    print(x)