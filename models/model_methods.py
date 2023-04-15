import pandas as pd
from gensim import corpora
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences
import gensim
import time
import torch
from gensim.models import TfidfModel
import os
import numpy as np

from sklearn.metrics import classification_report

import util
import constants


def make_fasttext_model(temp_df, sg=1, min_count=2, vector_size=300, workers=3, window=5, fasttext_file=None):
    ft_model = FastText(sg=sg, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                        sentences=temp_df)

    if fasttext_file is None:
        fasttext_file = constants.DATA_FOLDER + 'models/' + 'fasttext_' + str(vector_size) + "_" + '.model'
    ft_model.save(fasttext_file)

    return ft_model, fasttext_file


def make_fasttext_vector_cnn(w2v_model, sentence, device, max_sen_len):
    X = [0 * max_sen_len]
    i = 0
    for word in sentence:
        if word not in w2v_model.wv.key_to_index:
            print(i)
            X[i] = 0
        else:
            X[i] = w2v_model.wv.key_to_index[word]
            i += 1
    return torch.tensor(X, dtype=torch.long, device=device).view(1, -1)


def make_word2vec_model(temp_df, padding=True, sg=1, min_count=2, vector_size=300, workers=3, window=3, word2vec_file=None):
    if padding:
        # util.output(len(top_data_df_small))
        temp_df = pd.Series(temp_df).values
        temp_df = list(temp_df)
        temp_df.append(['pad'])
        if word2vec_file is None:
            word2vec_file = constants.DATA_FOLDER + 'models/' + 'word2vec_' + str(vector_size) + '_PAD.model'
    else:
        if word2vec_file is None:
            word2vec_file = constants.DATA_FOLDER + 'models/' + 'word2vec_' + str(vector_size) + '.model'
    w2v_model = Word2Vec(temp_df, min_count=min_count, vector_size=vector_size, workers=workers, window=window, sg=sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file

def make_vector_index_map(model, sentence, max_sen_len, is_fasttext):
    sentence_len = len(sentence)
    i = max_sen_len - sentence_len

    if not is_fasttext:
        padding_idx = model.key_to_index['pad']
        sentence_vec = [padding_idx for i in range(max_sen_len)]
    else:
        sentence_vec = [0] * max_sen_len

    for word in sentence:
        if word not in model.key_to_index:
            # in case of fasttext, we can use most similar vector
            if is_fasttext:
                sentence_vec[i] = model.key_to_index[model.most_similar(word)[0][0]]
            else:
                sentence_vec[i] = 0
        else:
            sentence_vec[i] = model.key_to_index[word]
        i += 1

    return sentence_vec



def make_word2vec_vector_cnn(model, sentence, device, max_sen_len, is_fasttext):
    if not is_fasttext:
        padding_idx = model.key_to_index['pad']
        sentence_vec = [padding_idx for i in range(max_sen_len)]
    else:
        sentence_vec = [0] * max_sen_len

    i = 0
    for word in sentence:
        if word not in model.key_to_index:
            if is_fasttext:
                sentence_vec[i] = model.key_to_index[model.most_similar(word)[0][0]]
            else:
                sentence_vec[i] = 0
        else:
            sentence_vec[i] = model.key_to_index[word]
        i += 1
    return torch.tensor(sentence_vec, dtype=torch.long, device=device).view(1, -1)


# Function to make bow vector to be used as input to network
def make_bow_vector(review_dict, sentence, device):
    vocab_size = len(review_dict)
    vec = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    for word in sentence:
        vec[review_dict.token2id[word]] += 1
    return vec.view(1, -1).float()


def create_tfidf_model_file(review_dict, df_sentiment, X_train, filename):
    # BOW corpus is required for tfidf model
    corpus = [review_dict.doc2bow(line) for line in df_sentiment['tokens']]

    # TF-IDF Model
    tfidf_model = TfidfModel(corpus)

    # start_time = time.time()
    # Storing the tfidf vectors for training data in a file
    vocab_len = len(review_dict.token2id)
    with open(filename, 'w+', encoding='utf-8') as tfidf_file:
        for index, row in X_train.items():
            doc = review_dict.doc2bow(row)
            features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                header = ";".join(str(review_dict[ele]) for ele in range(vocab_len))
                header = header.replace("\n", "")
                tfidf_file.write(header)
                tfidf_file.write("\n")
            line1 = ";".join([str(vector_element) for vector_element in features])
            tfidf_file.write(line1)
            tfidf_file.write('\n')
    # print("Time taken to create tfidf for :" + str(time.time() - start_time))

    return tfidf_model


def create_bow_model_file(review_dict, df_sentiment, X_train, filename):
    # start_time = time.time()
    vocab_len = len(review_dict)
    with open(filename, 'w+', encoding='utf-8') as bow_file:
        for index, row in X_train.items():
            features = gensim.matutils.corpus2csc([review_dict.doc2bow(row)],
                                                  num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                util.output("Header")
                header = ";".join(str(review_dict[ele]) for ele in range(vocab_len))
                header = header.replace("\n", "")
                bow_file.write(header)
                bow_file.write("\n")
            line1 = ";".join([str(vector_element) for vector_element in features])
            bow_file.write(line1)
            bow_file.write('\n')

    # util.output("Time taken to create bow for :" + str(time.time() - start_time))


# Function to return the dictionary either with padding word or without padding
def make_dict(top_data_df_small, padding=True):
    if padding:
        util.output("Dictionary with padded token added")
        review_dict = corpora.Dictionary([['pad']])
        review_dict.add_documents(top_data_df_small['tokens'])
    else:
        util.output("Dictionary without padding")
        review_dict = corpora.Dictionary(top_data_df_small['tokens'])
    return review_dict


def load_w2vec_model(data_df_ranked, word2vec_file):
    if constants.CREATE_MODEL:
        w2v_model, word2vec_file = make_word2vec_model(data_df_ranked)
    elif os.path.exists(word2vec_file):
        w2v_model = gensim.models.KeyedVectors.load(word2vec_file)
    else:
        util.exception("Word2vec model not found")

    return w2v_model, word2vec_file


def load_fasttext_model(data_df_ranked, fasttext_file):
    # fasttext_file = constants.DATA_FOLDER_TOK_STM + "fasttext_300_en.bin"
    if constants.CREATE_MODEL:
        ft_model, fasttext_file = make_fasttext_model(data_df_ranked['tokens'])
    elif os.path.exists(fasttext_file):
        ft_model = gensim.models.KeyedVectors.load(fasttext_file)
    else:
        ft_model, fasttext_file = make_fasttext_model(data_df_ranked['tokens'])
    return ft_model, fasttext_file


def testing_classificator_with_tfidf(clf_decision, tfidf_model, review_dict, X_test, Y_test_sentiment):
    test_features = []
    vocab_len = len(review_dict.token2id)

    # start_time = time.time()
    for index, row in X_test.items():
        doc = review_dict.doc2bow(row)
        features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf_decision.predict(test_features)
    util.output(classification_report(Y_test_sentiment, test_predictions))
    # print("Time taken to predict using TF-IDF:" + str(time.time() - start_time))


def testing_classificator_with_bow(clf_decision, review_dict, X_test, Y_test_sentiment):
    test_features = []
    vocab_len = len(review_dict.token2id)

    # start_time = time.time()
    for index, row in X_test.items():
        features = gensim.matutils.corpus2csc([review_dict.doc2bow(row)],
                                              num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf_decision.predict(test_features)
    util.output(classification_report(Y_test_sentiment, test_predictions))
    # print("Time taken to predict using TF-IDF:" + str(time.time() - start_time))
