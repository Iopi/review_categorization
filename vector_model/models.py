import pandas as pd
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences
import gensim
import time
import torch
from gensim.models import TfidfModel

import util
import constants


def make_fasttext_model(top_data_df_small, sg=1, min_count=1, vector_size=300, workers=3, window=5):
    temp_df = top_data_df_small['stemmed_tokens']
    ft_model = FastText(sg=sg, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                        sentences=temp_df)
    fasttext_file = constants.DATA_FOLDER + 'models/' + 'fasttext_' + str(vector_size) + '.model'
    ft_model.save(fasttext_file)

    return ft_model, fasttext_file


def make_fasttext_vector_cnn(w2v_model, sentence, device, max_sen_len):
    X = []
    i = 0
    for word in sentence:
        if word not in w2v_model.wv.key_to_index:
            print(i)
        else:
            X[i] = w2v_model.wv.key_to_index[word]
            i += 1
    return torch.tensor(X, dtype=torch.long, device=device).view(1, -1)


def make_word2vec_model(top_data_df_small, padding=True, sg=1, min_count=1, vector_size=500, workers=3, window=3):
    if padding:
        # util.output(len(top_data_df_small))
        temp_df = pd.Series(top_data_df_small['stemmed_tokens']).values
        temp_df = list(temp_df)
        temp_df.append(['pad'])
        word2vec_file = constants.DATA_FOLDER + 'models/' + 'word2vec_' + str(vector_size) + '_PAD.model'
    else:
        temp_df = top_data_df_small['stemmed_tokens']
        word2vec_file = constants.DATA_FOLDER + 'models/' + 'word2vec_' + str(vector_size) + '.model'
    w2v_model = Word2Vec(temp_df, min_count=min_count, vector_size=vector_size, workers=workers, window=window, sg=sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file


def make_word2vec_vector_cnn(w2v_model, sentence, device, max_sen_len):
    padding_idx = w2v_model.wv.key_to_index['pad']
    padded_X = [padding_idx for i in range(max_sen_len)]
    i = 0
    for word in sentence:
        if word not in w2v_model.wv.key_to_index:
            padded_X[i] = 0
            # util.output(word)
        else:
            padded_X[i] = w2v_model.wv.key_to_index[word]
        i += 1
    return torch.tensor(padded_X, dtype=torch.long, device=device).view(1, -1)


# Function to make bow vector to be used as input to network
def make_bow_vector(review_dict, sentence, device):
    vocab_size = len(review_dict)
    vec = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    for word in sentence:
        vec[review_dict.token2id[word]] += 1
    return vec.view(1, -1).float()


def create_tfidf_model_file(review_dict, df_sentiment, X_train, filename):
    # BOW corpus is required for tfidf model
    corpus = [review_dict.doc2bow(line) for line in df_sentiment['stemmed_tokens']]

    # TF-IDF Model
    tfidf_model = TfidfModel(corpus)

    # start_time = time.time()
    # Storing the tfidf vectors for training data in a file
    vocab_len = len(review_dict.token2id)
    with open(filename, 'w+') as tfidf_file:
        for index, row in X_train.iterrows():
            doc = review_dict.doc2bow(row['stemmed_tokens'])
            features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                header = ",".join(str(review_dict[ele]) for ele in range(vocab_len))
                util.output(header)
                util.output(tfidf_model[doc])
                tfidf_file.write(header)
                tfidf_file.write("\n")
            line1 = ",".join([str(vector_element) for vector_element in features])
            tfidf_file.write(line1)
            tfidf_file.write('\n')
    # print("Time taken to create tfidf for :" + str(time.time() - start_time))

    return tfidf_model

def create_bow_model_file(review_dict, df_sentiment, X_train, filename):
    # start_time = time.time()
    vocab_len = len(review_dict)
    with open(filename, 'w+') as bow_file:
        for index, row in X_train.iterrows():
            features = gensim.matutils.corpus2csc([review_dict.doc2bow(row['stemmed_tokens'])],
                                                  num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                util.output("Header")
                header = ",".join(str(review_dict[ele]) for ele in range(vocab_len))
                util.output(header)
                bow_file.write(header)
                bow_file.write("\n")
            line1 = ",".join([str(vector_element) for vector_element in features])
            bow_file.write(line1)
            bow_file.write('\n')

    # util.output("Time taken to create bow for :" + str(time.time() - start_time))