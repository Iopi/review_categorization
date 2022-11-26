import pandas as pd
import matplotlib.pyplot as plt
import time
import os.path
import numpy as np
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



def classification_sentiments(data_df_ranked, categories, binary):
    start_time = time.time()

    # load or create word2vec model
    # vec_model, vec_model_file = models.load_w2vec_model(data_df_ranked)
    # load or create fasttext model
    vec_model, vec_model_file = models.load_fasttext_model(data_df_ranked)

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
        X_train, X_test, Y_train, Y_test = preprocessing_methods.split_train_test(df_sentiment, category_name, test_size=0.25)

        # Plotting the sentiment distribution
        util.plot_category_distribution(Y_train, category_name)

        # Use cuda if present
        device = util.device_recognition()

        # X # LSTM with w2v/fasttext model
        max_sen_len = df_sentiment.stemmed_tokens.map(len).max()
        lstm_model = lstm.training_LSTM(vec_model, vec_model_file, device, max_sen_len, X_train, X_test,
                                   Y_train[category_name], Y_test[category_name], binary, batch_size=1)
        lstm.testing_LSTM(lstm_model, vec_model, device, max_sen_len, X_test['stemmed_tokens'], Y_test[category_name])

        # 1 # CNN with w2v/fasttext model
        # max_sen_len = df_sentiment.stemmed_tokens.map(len).max()
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


def create_model(lang):
    top_data_df = pd.read_excel(constants.DATA_FOLDER + 'feed_cs_model.xlsx', sheet_name="Sheet1")
    # Tokenize the text column to get the new column 'tokenized_text'

    # translator = Translator()
    # translated_text = translator.translate('ahoj, jak se máš?', src='cs', dest='en')
    # print(translated_text.text)

    preprocessing_methods.tokenization(top_data_df)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(top_data_df, lang)

    # creating model
    # models.make_word2vec_model(top_data_df, padding=False)
    # models.make_word2vec_model(top_data_df, padding=True)
    models.make_fasttext_model(top_data_df, lang)


def main():
    lang = 'cs'
    # only creating model
    # create_model(lang)

    # top_data_df = pd.read_excel(constants.DATA_FOLDER + constants.REVIEWS_DATA_NAME, sheet_name="Sheet1", nrows=550)
    top_data_df = pd.read_excel(constants.DATA_FOLDER + constants.REVIEWS_DATA_NAME, sheet_name="Sheet1")
    top_data_df = top_data_df.dropna(thresh=4)
    # util.output("Columns in the original dataset:\n")
    # util.output(top_data_df.columns)

    # Removing the stop words
    # preprocessing.remove_stopwords()

    # Tokenize the text column to get the new column 'tokenized_text'
    preprocessing_methods.tokenization(top_data_df)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(top_data_df, lang)

    classes = top_data_df.columns[1:10]

    temp_data = top_data_df.copy()
    # annotated 1, not annotated 0
    preprocessing_methods.map_sentiment_annotated(temp_data)
    classification_sentiments(temp_data, classes, True)
    #
    # temp_data = top_data_df.copy()
    # # positive 1, negative and neutral 0
    # map_sentiment_positive(temp_data)
    # classification_sentiments(temp_data, classes, True)
    # #
    # temp_data = top_data_df.copy()
    # # negative 1, positive and neutral 0
    # map_sentiment_negative(temp_data)
    # classification_sentiments(temp_data, classes)

    # temp_data = top_data_df.copy()
    # # neutral 0, positive 1 and negative 2
    # map_sentiment(temp_data)
    # classification_sentiments(temp_data, classes, False)

    # map_sentiment_annotate(top_data_df)
    # map_sentiment(top_data_df)


if __name__ == "__main__":
    main()
