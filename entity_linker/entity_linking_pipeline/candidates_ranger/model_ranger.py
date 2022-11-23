import re
import functools
import os
from typing import Any, List, Dict
from collections import OrderedDict

import fasttext as ft
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.config_utilities import load_config
from utils.paths import ENTITY_LINKER
from utils.normalize import normalize_mystem
from entity_linker.entity_linking_pipeline.candidates_ranger import BaseCandidatesRanger


class ModelRanger(BaseCandidatesRanger):

    def __init__(self, is_use_cosine=False, is_use_mean=False, model_path=None):
        config = load_config()
        self._is_use_cosine = is_use_cosine
        self._is_use_mean = is_use_mean
        self.ft_model = ft.load_model(config['entity_linker']['path_to_fasttext_model'])
        self._model_path =  os.path.join(ENTITY_LINKER, model_path)

    def range_candidates_set(self, candidates: List[Dict[str, Any]], context: List[str], **kwargs) -> Dict[str, int]:
        descs, ids = self._get_candidates_inf(candidates)
        result = self._get_model_prediction(descs, context, ids)
        sorted_candidates = OrderedDict(sorted(result, key=lambda x: x[1], reverse=True))
        return sorted_candidates

    def _get_model_prediction(self, descs, context, ids):
        context_vector = self._get_fasttext_vectors_for_phrase(' '.join(context))
        x_test = self._get_features(descs, context_vector)
        return self._get_model_probas(x_test, ids)

    def _get_model_probas(self, x_test, ids):
        model_result = [0] * len(ids)
        result = list()
        for i, res in enumerate(model_result):
            result.append((ids[i], res))
        return result

    def _get_candidates_inf(self, candidates):
        # тоже в utils можно отправить
        descs = list()
        ids = list()
        for candidate_dict in candidates:
            full_desc = candidate_dict['names']
            full_desc.extend([candidate_dict['desc']])
            descs.append(' '.join(full_desc))
            ids.append(candidate_dict['id'])
        return descs, ids

    def _get_features(self, descs, context_vector):
        if self._is_use_cosine:
            return self._get_features_concat_cosine(descs, context_vector)
        elif self._is_use_mean:
            return self._get_features_mean(descs, context_vector)
        else:
            return self._get_features_concat(descs, context_vector)

    def _get_features_concat(self, descs, context_vector):
        features = np.zeros((len(descs), 600))
        for i, desc in enumerate(descs):
            kb_vector = self._get_fasttext_vectors_for_phrase(' '.join(desc))
            features[i] = np.concatenate((context_vector, kb_vector), axis=0)
        return features

    def _get_features_concat_cosine(self, descs, context_vector):
        features = np.zeros((len(descs), 601))
        for i, desc in enumerate(descs):
            kb_vector = self._get_fasttext_vectors_for_phrase(' '.join(desc))
            cosine_vector = cosine_similarity([context_vector], [kb_vector])[0]
            features[i] = np.concatenate((context_vector, kb_vector, cosine_vector), axis=0)
        return features

    def _get_features_mean(self, descs, context_vector):
        features = np.zeros((len(descs), 900))
        for i, desc in enumerate(descs):
            kb_vector = self._get_fasttext_vectors_for_phrase(' '.join(desc))
            mean_value = (context_vector + kb_vector) / 2
            features[i] = np.concatenate((context_vector, kb_vector, mean_value), axis=0)
        return features

    @functools.lru_cache(maxsize=10000)
    def _get_fasttext_vectors_for_phrase(self, phrase: str):
        phrase = re.sub('[.,!?:;]', ' ', phrase)
        phrase = re.sub('  ', ' ', phrase)
        wordlist = normalize_mystem(phrase).split()
        sentence_vec = np.zeros((300,))
        number_of_words = len(wordlist)
        for word in wordlist:
            wordvec = self.ft_model.get_word_vector(str(word))
            if wordvec.any():
                sentence_vec += wordvec
            else:
                number_of_words -= 1
        if number_of_words == 0:
            return []
        else:
            return sentence_vec / number_of_words
