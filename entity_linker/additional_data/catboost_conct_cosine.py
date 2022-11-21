import csv
import os
import re
import functools

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import fasttext as ft
import numpy as np

from utils.config_utilities import load_config

config = load_config()
ft_model = ft.load_model(config['entity_linker']['path_to_fasttext_model'])


def train_catboost(dataset_directory):
    inputs, y = load_cloudscience(dataset_directory)
    x = get_features(inputs)
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, stratify=y, test_size=0.2)
    model = CatBoostClassifier(early_stopping_rounds=10, auto_class_weights='Balanced')
    model.fit(x_train, y_train, eval_set=(x_eval, y_eval))
    model.save_model('model_mean', format="cbm", export_parameters=None, pool=None)


def get_features_cosine(inputs):
    features = np.zeros((len(inputs), 601))
    for i, input_sample in enumerate(inputs):
        context_vector = get_fasttext_vectors_for_phrase(' '.join(input_sample[0]), ft_model)
        kb_vector = get_fasttext_vectors_for_phrase(' '.join(input_sample[1]), ft_model)
        cosine_vector = cosine_similarity([context_vector], [kb_vector])[0]
        features[i] = np.concatenate((context_vector, kb_vector, cosine_vector), axis=0)
    return features


def get_features(inputs):
    features = np.zeros((len(inputs), 900))
    for i, input_sample in enumerate(inputs):
        context_vector = get_fasttext_vectors_for_phrase(' '.join(input_sample[0]), ft_model)
        kb_vector = get_fasttext_vectors_for_phrase(' '.join(input_sample[1]), ft_model)
        mean_value = (context_vector + kb_vector) / 2
        features[i] = np.concatenate((context_vector, kb_vector, mean_value), axis=0)
    return features


def load_cloudscience(dataset_directory):
    result = list()
    labels = list()
    data_files = sorted(os.listdir(dataset_directory))
    for fname in data_files:
        if '.csv' not in fname:
            continue
        with open(os.path.join(dataset_directory, fname)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                labels.append(int(row['label']))
                full_desc = list(row['names'])
                full_desc.extend([row['desc']])
                result.append((row['context'], ' '.join(full_desc)))
    return result, labels


@functools.lru_cache(maxsize=10000)
def get_fasttext_vectors_for_phrase(phrase: str, model):
    phrase = re.sub('[.,!?:;]', ' ', phrase)
    phrase = re.sub('  ', ' ', phrase)
    wordlist = phrase.split()
    sentence_vec = np.zeros((300,))
    number_of_words = len(wordlist)
    for word in wordlist:
        wordvec = model.get_word_vector(str(word))
        if wordvec.any():
            sentence_vec += wordvec
        else:
            number_of_words -= 1
    if number_of_words == 0:
        return []
    else:
        return sentence_vec / number_of_words


train_catboost('/home/anastasia/PycharmProjects/entity_linking/entity_linker/dataset_for_el_models/processed')
