import os

import gensim
import pandas as pd
import torch
from gensim import corpora
from gensim.models import FastText
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from sklearn.metrics import classification_report

import constants
from view import app_output
from controller import preprocessing


def create_vector_model(args):
    """
    Creates vector model
    :param args: user arguments
    :return:
    """
    # load feed
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")
    # preprocessing
    result = preprocessing.lower_split(top_data_df, args.lang, check_lang=False)
    # remove stop words
    preprocessing.remove_bad_words(result, args.lang)
    # remove empty tokens
    result = [x for x in result if x != ['']]
    # save vector model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    if args.model_type == 'ft':
        make_fasttext_model(result, fasttext_file=args.model_path)
    elif args.model_type == 'w2v':
        make_word2vec_model(result, word2vec_file=args.model_path)
    else:
        app_output.exception(f"Model type {args.model_type} not found.")

    app_output.output(f"Model created in path {args.model_path}")



def make_fasttext_model(data_df, sg=1, min_count=2, vector_size=300, workers=3, window=5, fasttext_file=None):
    """
    Creates fasttext model
    :param data_df: data dataframe
    :param sg: training algorithm: if 1 skip-gram else CBOW
    :param min_count: the model ignores all words with total frequency lower than this
    :param vector_size: dimensionality of the word vectors
    :param workers: use these many worker threads to train the model
    :param window: the maximum distance between the current and predicted word within a sentence
    :param fasttext_file: filename
    :return: fasttext model, fasttext filename
    """
    ft_model = FastText(sg=sg, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                        sentences=data_df)

    if fasttext_file is None:
        fasttext_file = constants.DATA_FOLDER + 'vec_model/' + 'fasttext_' + str(vector_size) + "_" + '.bin'
    ft_model.save(fasttext_file)

    return ft_model, fasttext_file


def make_word2vec_model(data_df, padding=True, sg=1, min_count=2, vector_size=300, workers=3, window=3,
                        word2vec_file=None):
    """
    Creates word2vec model
    :param data_df: data dataframe
    :param padding: if padding is used
    :param sg: training algorithm: if 1 skip-gram else CBOW
    :param min_count: the model ignores all words with total frequency lower than this
    :param vector_size: dimensionality of the word vectors
    :param workers: use these many worker threads to train the model
    :param window: the maximum distance between the current and predicted word within a sentence
    :param word2vec_file: filename
    :return: word2vec model, word2vec filename
    """
    if padding:
        data_df = pd.Series(data_df).values
        data_df = list(data_df)
        data_df.append(['pad'])
        if word2vec_file is None:
            word2vec_file = constants.DATA_FOLDER + 'vec_model/' + 'word2vec_' + str(vector_size) + '_PAD.bin'
    else:
        if word2vec_file is None:
            word2vec_file = constants.DATA_FOLDER + 'vec_model/' + 'word2vec_' + str(vector_size) + '.bin'
    w2v_model = Word2Vec(data_df, min_count=min_count, vector_size=vector_size, workers=workers, window=window, sg=sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file


def make_vector_index_map(model, sentence, max_sen_len, is_fasttext):
    """
    Makes vector index map
    :param model: vector model
    :param sentence: text to be mapped
    :param max_sen_len: maximal text length in data
    :param is_fasttext: if model is fasttext
    :return: index map
    """
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


def make_vector_index_map_cnn(model, sentence, device, max_sen_len, is_fasttext):
    """
    Makes vector index map for cnn
    :param model: vector model
    :param sentence: text to be mapped
    :param device: device (cpu/gpu)
    :param max_sen_len: maximal text length in data
    :param is_fasttext: if model is fasttext
    :return: index map
    """
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


def create_tfidf_model_file(review_dict, data_df, X_train, filename):
    """
    Creates tfidf model saved to file
    :param review_dict: reviews dictionary
    :param data_df: data dataframe
    :param X_train: training tokens
    :param filename: model filename
    :return: tfidf model
    """
    # BOW corpus is required for tfidf model
    corpus = [review_dict.doc2bow(line) for line in data_df['tokens']]
    # TF-IDF Model
    tfidf_model = TfidfModel(corpus)
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

    return tfidf_model


def create_bow_model_file(review_dict, data_df, X_train, filename):
    """
    Creates bow model saved to file
    :param review_dict: reviews dictionary
    :param data_df: data dataframe
    :param X_train: training tokens
    :param filename: model filename
    :return: bow model
    """
    vocab_len = len(review_dict)
    with open(filename, 'w+', encoding='utf-8') as bow_file:
        for index, row in X_train.items():
            features = gensim.matutils.corpus2csc([review_dict.doc2bow(row)],
                                                  num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                header = ";".join(str(review_dict[ele]) for ele in range(vocab_len))
                header = header.replace("\n", "")
                bow_file.write(header)
                bow_file.write("\n")
            line1 = ";".join([str(vector_element) for vector_element in features])
            bow_file.write(line1)
            bow_file.write('\n')


# Function to return the dictionary either with padding word or without padding
def make_dict(data_df, padding=True):
    """
    Creates dictionary from tokens
    :param data_df: data dataframe
    :param padding: if padding is used
    :return: dictionary from tokens
    """
    if padding:
        review_dict = corpora.Dictionary([['pad']])
        review_dict.add_documents(data_df['tokens'])
    else:
        review_dict = corpora.Dictionary(data_df['tokens'])
    return review_dict


def testing_classifier_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test):
    """
    Testing of standard classifier with tfidf model
    :param clf: classifier
    :param tfidf_model: tfidf model
    :param review_dict: dictionary from tokens
    :param X_test: test tokens
    :param Y_test: test labels
    :return:
    """
    test_features = []
    vocab_len = len(review_dict.token2id)

    for index, row in X_test.items():
        doc = review_dict.doc2bow(row)
        features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf.predict(test_features)
    app_output.output(classification_report(Y_test, test_predictions))


def testing_classifier_with_bow(clf, review_dict, X_test, Y_test):
    """
    Testing of standard classifier with bow model
    :param clf: classifier
    :param review_dict: dictionary from tokens
    :param X_test: test tokens
    :param Y_test: test labels
    :return:
    """
    test_features = []
    vocab_len = len(review_dict.token2id)

    for index, row in X_test.items():
        features = gensim.matutils.corpus2csc([review_dict.doc2bow(row)],
                                              num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf.predict(test_features)
    app_output.output(classification_report(Y_test, test_predictions))
