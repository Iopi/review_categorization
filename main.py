import pandas as pd
import argparse
import matplotlib.pyplot as plt
import time
import re
import os.path
import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch

import seaborn as sns
from googletrans import Translator

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch.nn.functional as F

from classifires import log_reg, decision_tree, svm, ffnn, cnn, lstm
from classifires.cnn import ConvolutionalNeuralNetworkClassifier
from classifires.ffnn import FeedforwardNeuralNetModel
from classifires.log_reg import LogisticRegressionClassifier
from classifires.lstm import LongShortTermMemory
from preprocessing import preprocessing_methods, czech_stemmer
from vector_model import models
import util
import constants

from gensim import corpora
import gensim
import fasttext


def classification_sentiments(data_df_ranked, categories, binary, args, test_data_df=None):
    start_time = time.time()

    # load or create word2vec model
    model_filename = None
    vector_filename = None
    # # load or create word2vec model
    # vec_model, model_filename = models.load_w2vec_model(data_df_ranked, args.model_path)
    # vec_model = vec_model.wv
    # # load or create fasttext model
    # vec_model, model_filename = models.load_fasttext_model(data_df_ranked, args.model_path)
    # vec_model = vec_model.wv
    model_filename = args.model_path
    vec_model_train = gensim.models.KeyedVectors.load(model_filename)
    vec_model_train = vec_model_train.wv
    # vector_filename = args.model_path
    # vec_model_train = KeyedVectors.load_word2vec_format(vector_filename, binary=False)
    if args.action == 'cross':
        vec_model_test = args.model_path_test
    else:
        vec_model_test = vec_model_train



    for category_name in categories:
        # if category_name != 'Staff':
        #     continue
        start_time_class = time.time()

        util.output("Classification sentiment " + category_name)

        # drop not needed rows
        if binary:
            df_sentiment = data_df_ranked[data_df_ranked[category_name] != 2]
        else:
            df_sentiment = data_df_ranked[data_df_ranked[category_name] != 3]

        # Plotting the sentiment distribution
        # plot_distribution(df_sentiment, category_name)

        # After selecting top few samples of each sentiment

        # util.output("After segregating and taking equal number of rows for each sentiment:")
        # util.output(df_sentiment[category_name].value_counts())
        # util.output(df_sentiment.head(10))

        # Call the train_test_split
        if test_data_df is None:
            X_train, X_test, Y_train, Y_test = preprocessing_methods.split_train_test(df_sentiment, category_name,
                                                                                      test_size=0.25)
        else:
            X_train = df_sentiment['tokens']
            Y_train = df_sentiment[category_name]
            X_test = test_data_df['tokens']
            Y_test = test_data_df[category_name]

        # Plotting the sentiment distribution
        util.plot_category_distribution(Y_train, category_name)

        # Use cuda if present
        device = util.device_recognition()

        # X # LSTM with w2v/fasttext model
        max_sen_len = df_sentiment.tokens.map(len).max()
        lstm_model = lstm.training_LSTM(vec_model_train, device, max_sen_len, X_train, X_test, Y_train, Y_test, binary,
                                        batch_size=1, model_filename=model_filename, vector_filename=vector_filename)
        lstm.testing_LSTM(lstm_model, vec_model_test, device, max_sen_len, X_test, Y_test)
        # lstm.testing_LSTM(lstm_model, vec_model.wv, device, max_sen_len, X_test, Y_test)

        # 1 # CNN with w2v/fasttext model
        # max_sen_len = df_sentiment.tokens.map(len).max()
        # cnn_model = cnn.training_CNN(vec_model, vec_model_file, device, max_sen_len, X_train, Y_train[category_name],
        #                          binary, padding=True)
        # cnn.testing_CNN(cnn_model, vec_model_file, vec_model, device, max_sen_len, X_test, Y_test[category_name])

        # 2 # FFNN
        # Make the dictionary without padding for the basic models
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # ff_nn_bow_model, ffnn_loss_file_name = ffnn.training_FFNN(review_dict, device, X_train, Y_train[category_name])
        # ffnn.testing_FFNN(review_dict, ff_nn_bow_model, ffnn_loss_file_name, device, X_test, Y_test[category_name])

        # 3 # Logistic Regresion with BoW model
        # util.output("Logistic Regresion - Bow - pytorch")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # bow_nn_model = log_reg.training_LogReg(review_dict, device, X_train, Y_train[category_name])
        # log_reg.testing_LogReg(review_dict, bow_nn_model, device, X_test, Y_test[category_name])

        # # 5 # Decision Tree with BoW model
        # util.output("Decision Tree - Bow")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = decision_tree.training_Decision_Tree(Y_train[category_name], filename)
        # models.testing_classificator_with_bow(clf, review_dict, X_test, Y_test[category_name])

        # # 4 # Decision Tree with Tfidf model
        # util.output("Decision Tree - Tfidf")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_tfidf.csv'
        # tfidf_model = models.create_tfidf_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = decision_tree.training_Decision_Tree(Y_train[category_name], filename)
        # models.testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test[category_name])
        #
        # # 6 # Linear SVM with BoW model
        # util.output("Linear SVM - BoW")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = svm.training_Linear_SVM(Y_train[category_name], filename)
        # models.testing_classificator_with_bow(clf, review_dict, X_test, Y_test[category_name])
        #
        # # 7 # Linear SVM with Tfidf model
        # util.output("Linear SVM - Tfidf")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_tfidf.csv'
        # tfidf_model = models.create_tfidf_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = svm.training_Linear_SVM(Y_train[category_name], filename)
        # models.testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test[category_name])
        #
        # # 8 # Logistic Regression with BoW model
        # util.output("Logistic Regression - Bow")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = log_reg.training_Logistic_Regression(Y_train[category_name], filename)
        # models.testing_classificator_with_bow(clf, review_dict, X_test, Y_test[category_name])
        #
        # # 9 # Logistic Regression with Tfidf model
        # util.output("Logistic Regression - Tfidf")
        # review_dict = models.make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # tfidf_model = models.create_tfidf_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = log_reg.training_Logistic_Regression(Y_train[category_name], filename)
        # models.testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test[category_name])

        util.output("Time taken to predict " + category_name + " :" + str(time.time() - start_time_class))
        # break
    util.output("Time taken to predict all:" + str(time.time() - start_time))


def create_unsup_lower_split_model(args):
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")

    top_data_df['Text'] = top_data_df['Text'].str.lower()

    with open(args.feed_path[:-4] + 'txt', 'w', encoding="utf-8") as f:
        index = 0
        for line in top_data_df['Text'].values:
            index += 1

            try:
                line = re.sub('([.,!?()])', r' \1 ', line)
                line = re.sub('\s{2,}', ' ', line)
            except:
                continue

            f.write(line)
            f.write('\n')

    model = fasttext.train_unsupervised(args.feed_path[:-4] + 'txt', model='skipgram', dim=300)
    # filename = constants.DATA_FOLDER_UNSUPERVISED + "fasttext_unsup_" + args.lang + ".bin"
    filename = args.model_path
    model.save_model(filename)

    # util.bin2vec(filename)


def create_lower_split_model(args):
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")
    # lowercase and split .,?!
    result = preprocessing_methods.lower_split(top_data_df)

    models.make_fasttext_model(result, fasttext_file=constants.DATA_FOLDER + 'lower_split_models/' + 'fasttext_300_' +
                                                     args.lang + ".bin")



def create_token_stem_model(args):
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")

    # Tokenize the text column to get the new column 'tokenized_text'
    top_data_df = preprocessing_methods.tokenization(top_data_df, args.lang)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(top_data_df, args.lang)

    # creating model
    # models.make_word2vec_model(top_data_df, padding=False)
    # models.make_word2vec_model(top_data_df, padding=True)
    models.make_fasttext_model(top_data_df['tokens'], fasttext_file=args.model_path)


def parse_agrs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-mp', dest='model_path', type=str, help='Destination of model for classification for train'
                                                                 ' (and test for mono-lingual classification).')
    parser.add_argument('-rp', dest='reviews_path', type=str, help='Destination of reviews for classification for'
                                                                   'train (and test for mono-lingual classification).')
    parser.add_argument('-fp', dest='feed_path', type=str, help='Destination of feed for model creating.')
    parser.add_argument('-mptest', dest='model_path_test', type=str,
                        help='Destination of model for cross-lingual classification for test.')
    parser.add_argument('-rptest', dest='reviews_path_test', type=str,
                        help='Destination of test reviews for classification for test.')
    parser.add_argument('-l', dest='lang', help='Language of train reviews (and test for mono-lingual classification).')
    parser.add_argument('-ltest', dest='lang_test', help='Language of test reviews.')
    parser.add_argument('-a', dest='action', help="Action of application. 'mono' mono-lingual classification, 'cross' "
                                                  "cross-lingual classification and 'model' create model to folder "
                                                  "added to 'model_filename' destination.", default='mono')

    args = parser.parse_args()

    if args.model_path is None:
        raise Exception("Model path 'model_path' must be set.")
    if args.action == 'model':
        if args.feed_path is None:
            raise Exception("Feed path 'feed_path' must be set.")
    else:
        if args.reviews_path is None:
            raise Exception("Reviews path 'reviews_path' must be set.")
        if args.lang is None:
            raise Exception("Language 'lang' must be set.")
        if args.action == 'cross':
            if args.model_path_test is None:
                raise Exception("Model path for test 'model_path_test' must be set.")
            if args.reviews_path_test is None:
                raise Exception("Reviews path for test 'reviews_path_test' must be set.")
            if args.lang_test is None:
                raise Exception("Language for test 'lang_test' must be set.")
        elif args.action != 'mono':
            raise Exception(
                "Wrong argument for action. Action of application. 'mono' mono-lingual classification, 'cross' "
                "cross-lingual classification and 'model' create model to folder "
                "added to 'model_filename' destination.")

    return args


def classification_sentiments_annotated(reviews_df, reviews_test_df, classes, args):
    # annotated 1, not annotated 0
    temp_data = reviews_df.copy()
    preprocessing_methods.map_sentiment_annotated(temp_data)
    if reviews_test_df:
        preprocessing_methods.map_sentiment_annotated(reviews_test_df)
        classification_sentiments(temp_data, classes, True, args, reviews_test_df)
    classification_sentiments(temp_data, classes, True, args)


def classification_sentiments_positive(reviews_df, reviews_test_df, classes, args):
    # positive 1, negative and neutral 0
    temp_data = reviews_df.copy()
    preprocessing_methods.map_sentiment_positive(temp_data)
    if reviews_test_df:
        preprocessing_methods.map_sentiment_positive(reviews_test_df)
        classification_sentiments(temp_data, classes, True, args, reviews_test_df)
    classification_sentiments(temp_data, classes, True, args)


def classification_sentiments_negative(reviews_df, reviews_test_df, classes, args):
    # negative 1, positive and neutral 0
    temp_data = reviews_df.copy()
    preprocessing_methods.map_sentiment_negative(temp_data)
    if reviews_test_df:
        preprocessing_methods.map_sentiment_negative(reviews_test_df)
        classification_sentiments(temp_data, classes, True, args, reviews_test_df)
    classification_sentiments(temp_data, classes, True, args)


def main():
    args = parse_agrs()

    reviews_test_df = None
    # only creating model
    if args.action == 'model':
        create_token_stem_model(args)
        # create_lower_split_model(args)
        # create_unsup_lower_split_model(args)
        exit(0)
    elif args.action == 'cross':
        reviews_test_df = pd.read_excel(args.reviews_path_test, sheet_name="Sheet1")
        reviews_test_df = reviews_test_df.dropna(thresh=4)
        reviews_test_df['tokens'] = preprocessing_methods.lower_split(reviews_test_df)

    # top_data_df = pd.read_excel(constants.DATA_FOLDER + constants.REVIEWS_DATA_NAME, sheet_name="Sheet1", nrows=550)
    reviews_df = pd.read_excel(args.reviews_path, sheet_name="Sheet1")
    reviews_df = reviews_df.dropna(thresh=4)

    # util.output("Columns in the original dataset:\n")
    # util.output(top_data_df.columns)

    # Removing the stop words
    # preprocessing.remove_stopwords()
    #
    # Tokenize the text column to get the new column 'tokenized_text'
    preprocessing_methods.tokenization(reviews_df)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(reviews_df, args.lang)

    # reviews_df['tokens'] = preprocessing_methods.lower_split(reviews_df)

    classes = reviews_df.columns[1:10]

    classification_sentiments_annotated(reviews_df, reviews_test_df, classes, args)
    classification_sentiments_positive(reviews_df, reviews_test_df, classes, args)
    classification_sentiments_negative(reviews_df, reviews_test_df, classes, args)


    # temp_data = top_data_df.copy()
    # # neutral 0, positive 1 and negative 2
    # map_sentiment(temp_data)
    # classification_sentiments(temp_data, classes, False)

    # map_sentiment_annotate(top_data_df)
    # map_sentiment(top_data_df)


if __name__ == "__main__":
    main()
