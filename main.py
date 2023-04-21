import argparse
import time

import pandas as pd
from gensim.models import KeyedVectors

import constants
import util
from classifires import lstm, cnn, svm, log_reg, decision_tree
from models import model_methods
from preprocessing import preprocessing_methods
from transformation import transformation


def parse_agrs():
    parser = argparse.ArgumentParser(description='Reviews classification.')
    parser.add_argument('-mp', dest='model_path', type=str, help='Destination of vector model for classification for '
                                                                 'train (and test for mono-lingual classification).')
    parser.add_argument('-rp', dest='reviews_path', type=str, help='Destination of reviews for classification for'
                                                                   'train (and test for mono-lingual classification).')
    parser.add_argument('-fp', dest='feed_path', type=str, help='Destination of feed for model creating.')
    parser.add_argument('-mptest', dest='model_path_test', type=str,
                        help='Destination of vector model for cross-lingual classification for test.')
    parser.add_argument('-mt', dest='model_type', type=str, help="Type of vector model. 'ft' fasttext model, 'w2v' "
                                                                 "word2vec model, 'tfidf' tf-iff model, 'bow' bag of words model")
    parser.add_argument('-cm', dest='classi_model', type=str, help="Type of classification model. 'lstm' lstm model, "
                                                                   "'cnn' cnn model, 'svm' svm model, 'logreg' logistic "
                                                                   "regression model, 'dectree' decision tree model")
    parser.add_argument('-rptest', dest='reviews_path_test', type=str,
                        help='Destination of test reviews for classification for test.')
    parser.add_argument('-l', dest='lang', help='Language of train reviews (and test for mono-lingual classification).')
    parser.add_argument('-ltest', dest='lang_test', help='Language of test reviews.')
    parser.add_argument('-a', dest='action', help="Action of application. 'mono' mono-lingual classification, "
                                                  "'monotest' mono-lingual classification with separate files for "
                                                  "train and test, 'cross' cross-lingual classification, 'translate' "
                                                  "translate classification and 'model' create model to folder added "
                                                  "to 'model_filename' destination.")

    args = parser.parse_args()
    util.output(args)

    if args.model_type is None:
        util.exception("Model type 'model_type' must be set.")
    if args.model_type == 'w2v' or args.model_type == 'ft':
        if args.model_path is None:
            util.exception("Model path 'model_path' must be set.")
    if args.lang is None:
        util.exception("Language 'lang' must be set.")
    if args.action == 'model':
        if args.feed_path is None:
            util.exception("Feed path 'feed_path' must be set.")
    else:
        if args.classi_model is None:
            util.exception("Classification model 'classi_model' must be set.")
        if args.reviews_path is None:
            util.exception("Reviews path 'reviews_path' must be set.")
        if args.action == 'cross':
            if args.model_path_test is None:
                util.exception("Model path for test 'model_path_test' must be set.")
            if args.reviews_path_test is None:
                util.exception("Reviews path for test 'reviews_path_test' must be set.")
            if args.lang_test is None:
                util.exception("Language for test 'lang_test' must be set.")
        elif args.action == 'monotest' or args.action == 'translate':
            if args.reviews_path_test is None:
                util.exception("Reviews path for test 'reviews_path_test' must be set.")
            if args.lang_test is None:
                util.exception("Language for test 'lang_test' must be set.")
        elif args.action != 'mono':
            util.exception(
                "Wrong argument for action. Action of application. 'mono' mono-lingual classification, 'cross' "
                "cross-lingual classification and 'model' create model to folder "
                "added to 'model_filename' destination.")

        if ((args.model_type == 'lstm' or args.model_type == 'cnn') and (args.classi_model == 'bow' or
                                                                         args.classi_model == 'tfidf')) or (
                (args.model_type == 'svm' or args.model_type == 'logreg' or
                 args.model_type == 'dectree') and (args.classi_model == 'w2v' or
                                                    args.classi_model == 'ft')):
            util.exception("Neural network classifiers (lstm, cnn) can only be in combination with text representation "
                           "word2vec and fasttext and other classifiers (Support vector machines, Logistic regression "
                           "and Decision tree) can only be in combination with bow and tf-idf text representation")
    return args


def classification_sentiments(data_df_ranked, categories, model_tuple, args, test_data_df=None):
    start_time = time.time()

    model_filename_train = model_tuple[0]
    model_filename_test = model_tuple[1]
    vec_model_train = model_tuple[2]
    vec_model_test = model_tuple[3]
    trans_matrix = model_tuple[4]
    is_fasttext = model_tuple[5]

    # Use cuda if present
    device = util.device_recognition()

    util.print_info(args, is_fasttext)

    df_test = None
    for category_name in categories:

        start_time_class = time.time()

        util.output("Classification sentiment " + category_name)

        # drop not needed rows
        df_sentiment = data_df_ranked[data_df_ranked[category_name] != 2]

        # Plotting the sentiment distribution
        # util.plot_distribution(df_sentiment, category_name)

        util.sentiment_count(df_sentiment, category_name)

        # Call the train_test_split
        if test_data_df is None:
            X_train, X_test, Y_train, Y_test = preprocessing_methods.split_train_test(df_sentiment, category_name,
                                                                                      test_size=0.20)
        else:
            df_test = test_data_df[test_data_df[category_name] != 2]

            X_train = df_sentiment['tokens']
            Y_train = df_sentiment[category_name]
            X_test = df_test['tokens']
            Y_test = df_test[category_name]

        util.compute_majority_class(Y_test)
        # if simple classifier
        if vec_model_train is None:
            clf = None
            filename = None
            tfidf_model = None
            review_dict = model_methods.make_dict(df_sentiment, padding=False)
            # create text representation
            # bag of words vector representation
            if args.model_type == 'bow':
                filename = constants.MODEL_FOLDER + 'train_review_bow.csv'
                model_methods.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
                clf = decision_tree.training_Decision_Tree(Y_train, filename)

            # tf-idf vector representation
            elif args.model_type == 'tfidf':
                filename = constants.MODEL_FOLDER + 'train_review_tfidf.csv'
                tfidf_model = model_methods.create_tfidf_model_file(review_dict, df_sentiment, X_train, filename)
            else:
                util.exception(f"Wrong model type {args.model_type}")

            # train classificator
            # svm classificator
            if args.classi_model == "svm":
                clf = svm.training_Linear_SVM(Y_train, filename)

            # logistic regression classificator
            elif args.classi_model == "logreg":
                clf = log_reg.training_Logistic_Regression(Y_train, filename)

            # decision tree classificator
            elif args.classi_model == "dectree":
                clf = decision_tree.training_Decision_Tree(Y_train, filename)

            else:
                util.exception(f"Wrong classification model {args.classi_model}")

            # test classificator
            if args.model_type == 'bow':
                model_methods.testing_classificator_with_bow(clf, review_dict, X_test, Y_test)

            elif args.model_type == 'tfidf':
                model_methods.testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test)

        # if neural network
        else:
            max_sen_len = df_sentiment.tokens.map(len).max()
            if df_test is not None:
                max_sen_len_test = df_test.tokens.map(len).max()
                max_sen_len = max(max_sen_len, max_sen_len_test)

            # lstm classificator
            if args.classi_model == "lstm":
                lstm_model = lstm.training_LSTM(vec_model_train, trans_matrix, device, max_sen_len, X_train, Y_train,
                                                is_fasttext,
                                                batch_size=1, model_filename_train=model_filename_train,
                                                model_filename_test=model_filename_test)
                lstm.testing_LSTM(lstm_model, vec_model_test, device, max_sen_len, X_test, Y_test, is_fasttext)

            # cnn classificator
            elif args.classi_model == "cnn":
                cnn_model = cnn.training_CNN(vec_model_train, model_filename_train, trans_matrix, device, max_sen_len,
                                             X_train, Y_train,
                                             is_fasttext, padding=True, model_filename_test=model_filename_test)
                cnn.testing_CNN(cnn_model, vec_model_test, device, max_sen_len, X_test, Y_test, is_fasttext)
            else:
                util.exception(f"Wrong classification model {args.classi_model}")

        util.output("Time taken to predict " + category_name + " :" + (str(time.time() - start_time_class) + "\n"))
        # break
    util.output("Time taken to predict all:" + str(time.time() - start_time))


def classification_category_existence(reviews_df, reviews_test_df, classes, model_tuple, args):
    # annotated 1, not annotated 0
    util.output("Annotated 1, not annotated 0")
    temp_data = reviews_df.copy()
    preprocessing_methods.map_category_existence(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing_methods.map_category_existence(temp_data_test)
        classification_sentiments(temp_data, classes, model_tuple, args, temp_data_test)
    else:
        classification_sentiments(temp_data, classes, model_tuple, args)


def classification_sentiment(reviews_df, reviews_test_df, classes, model_tuple, args):
    # positive 1, negative 0
    util.output("Positive 1, negative 0")
    temp_data = reviews_df.copy()
    preprocessing_methods.map_sentiment(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing_methods.map_sentiment(temp_data_test)
        classification_sentiments(temp_data, classes, model_tuple, args, temp_data_test)
    else:
        classification_sentiments(temp_data, classes, model_tuple, args)


def load_models_and_trans_matrix(args, trans_method, filename):
    model_filename_train = None
    vec_model_train = None
    vec_model_test = None
    model_filename_test = None
    is_fasttext = None
    trans_matrix = None
    if args.model_type == 'ft':
        is_fasttext = True
    elif args.model_type == 'w2v':
        is_fasttext = False
    else:
        return model_filename_train, model_filename_test, vec_model_train, vec_model_test, trans_matrix, is_fasttext

    model_filename_train = args.model_path
    vec_model_train = KeyedVectors.load(model_filename_train)
    vec_model_train = vec_model_train.wv
    if args.action == 'cross':
        model_filename_test = args.model_path_test
        vec_model_test = KeyedVectors.load(model_filename_test)
        vec_model_test = vec_model_test.wv
        trans_matrix = transformation.get_trans_matrix(vec_model_train, vec_model_test, args.lang, args.lang_test,
                                                       trans_method, filename)
    else:
        vec_model_test = vec_model_train

    return model_filename_train, model_filename_test, vec_model_train, vec_model_test, trans_matrix, is_fasttext


def main():
    args = parse_agrs()

    reviews_test_df = None
    # only creating model
    if args.action == 'model':
        model_methods.create_lower_split_model(args)
        exit(0)
    elif args.action == 'cross' or args.action == 'translate' or args.action == 'monotest':
        reviews_test_df = pd.read_excel(args.reviews_path_test, sheet_name="Sheet1")
        reviews_test_df = reviews_test_df.dropna(thresh=4)
        if args.action == 'translate':
            preprocessing_methods.translate_data(reviews_test_df, args.lang_test, args.lang)
            reviews_test_df['tokens'] = preprocessing_methods.lower_split(reviews_test_df, args.lang)
            preprocessing_methods.remove_bad_words(reviews_test_df['tokens'], args.lang)

        else:
            reviews_test_df['tokens'] = preprocessing_methods.lower_split(reviews_test_df, args.lang_test)
            preprocessing_methods.remove_bad_words(reviews_test_df['tokens'], args.lang_test)

        reviews_test_df = reviews_test_df[reviews_test_df['tokens'].apply(lambda x: x != [''])]

    reviews_df = pd.read_excel(args.reviews_path, sheet_name="Sheet1")
    reviews_df = reviews_df.dropna(thresh=4)

    reviews_df['tokens'] = preprocessing_methods.lower_split(reviews_df, args.lang)
    preprocessing_methods.remove_bad_words(reviews_df['tokens'], args.lang)
    reviews_df = reviews_df[reviews_df['tokens'].apply(lambda x: x != [''])]

    classes = reviews_df.columns[1:10]

    trans_method, filename = "orto", constants.DICT_FOLDER + f"{args.lang}-{args.lang_test}_muj.txt"
    model_tuple = load_models_and_trans_matrix(args, trans_method, filename)

    classification_category_existence(reviews_df, reviews_test_df, classes, model_tuple, args)
    classification_sentiment(reviews_df, reviews_test_df, classes, model_tuple, args)


if __name__ == "__main__":
    main()
