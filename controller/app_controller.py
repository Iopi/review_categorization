import time
import pandas as pd
from gensim.models import KeyedVectors

import constants
from controller import preprocessing, transformation, vector_reprezentation
from model.classifiers import svm, lstm, cnn, log_reg, decision_tree
from view import app_output
import util


def classification(df_train, categories, model_tuple, args, df_test=None):
    """
    Detection of sentiment or detection existence for all categories
    :param df_train: train data
    :param categories: categories
    :param model_tuple: vector models, vector model filenames, transformation matrix and bool if model are fasttext model
    :param args: user arguments
    :param df_test: test data if exists
    :return:
    """

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

    df_test_category = None
    for category_name in categories:

        start_time_class = time.time()

        app_output.output("-------------------------------------")
        app_output.output("Classification sentiment " + category_name)

        # drop not needed rows
        df_train_category = df_train[df_train[category_name] != 2]

        # Plotting the sentiment distribution
        # util.plot_distribution(df_train_category, category_name)

        # util.sentiment_count(df_train_category, category_name)

        # split train data in case of monolingual classification
        if df_test is None:
            X_train, X_test, Y_train, Y_test = preprocessing.split_train_test(df_train_category, category_name,
                                                                              test_size=0.20)
        else:
            df_test_category = df_test[df_test[category_name] != 2]

            # util.sentiment_count(df_test_category, category_name)

            X_train = df_train_category['tokens']
            Y_train = df_train_category[category_name]
            X_test = df_test_category['tokens']
            Y_test = df_test_category[category_name]

        util.compute_majority_class(Y_test)
        app_output.output("-------------------------------------")

        # if simple classifier
        if vec_model_train is None:
            clf = None
            filename = None
            tfidf_model = None
            review_dict = vector_reprezentation.make_dict(df_train_category, padding=False)
            # create text representation
            # bag of words vector representation
            if args.model_type == 'bow':
                filename = constants.MODEL_FOLDER + 'train_review_bow.csv'
                vector_reprezentation.create_bow_model_file(review_dict, df_train_category, X_train, filename)
                clf = decision_tree.training_Decision_Tree(Y_train, filename)

            # tf-idf vector representation
            elif args.model_type == 'tfidf':
                filename = constants.MODEL_FOLDER + 'train_review_tfidf.csv'
                tfidf_model = vector_reprezentation.create_tfidf_model_file(review_dict, df_train_category, X_train,
                                                                            filename)
            else:
                app_output.exception(f"Wrong model type {args.model_type}")

            # train classifier
            # svm classifier
            if args.classi_model == "svm":
                clf = svm.training_Linear_SVM(Y_train, filename)

            # logistic regression classifier
            elif args.classi_model == "logreg":
                clf = log_reg.training_Logistic_Regression(Y_train, filename)

            # decision tree classifier
            elif args.classi_model == "dectree":
                clf = decision_tree.training_Decision_Tree(Y_train, filename)

            else:
                app_output.exception(f"Wrong classification model {args.classi_model}")

            # test classifier
            if args.model_type == 'bow':
                vector_reprezentation.testing_classifier_with_bow(clf, review_dict, X_test, Y_test)

            elif args.model_type == 'tfidf':
                vector_reprezentation.testing_classifier_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test)

        # if neural network
        else:
            max_sen_len = df_train_category.tokens.map(len).max()
            if df_test_category is not None:
                max_sen_len_test = df_test_category.tokens.map(len).max()
                max_sen_len = max(max_sen_len, max_sen_len_test)

            # lstm classifier
            if args.classi_model == "lstm":
                lstm_model = lstm.training_LSTM(vec_model_train, trans_matrix, device, max_sen_len, X_train, Y_train,
                                                is_fasttext,
                                                batch_size=1, model_filename_train=model_filename_train,
                                                model_filename_test=model_filename_test)
                lstm.testing_LSTM(lstm_model, vec_model_test, device, max_sen_len, X_test, Y_test, is_fasttext)

            # cnn classifier
            elif args.classi_model == "cnn":
                cnn_model = cnn.training_CNN(vec_model_train, model_filename_train, trans_matrix, device, max_sen_len,
                                             X_train, Y_train,
                                             is_fasttext, padding=True, model_filename_test=model_filename_test)
                cnn.testing_CNN(cnn_model, vec_model_test, device, max_sen_len, X_test, Y_test, is_fasttext)
            else:
                app_output.exception(f"Wrong classification model {args.classi_model}")

        app_output.output("Time taken to predict " + category_name + " :" + (str(time.time() - start_time_class) + "\n"))
    app_output.output("Time taken to predict all:" + str(time.time() - start_time))


def classification_category_existence(reviews_df, reviews_test_df, categories, model_tuple, args):
    """
    Prepare for classification of category existence
    :param reviews_df: train reviews
    :param reviews_test_df: test reviews
    :param categories: categories
    :param model_tuple: vector models, vector model filenames, transformation matrix and bool if model are fasttext model
    :param args: user arguments
    :return:
    """
    # annotated 1, not annotated 0
    app_output.output("-----------------------------\nAnnotated 1, not annotated 0\n-----------------------------")
    temp_data = reviews_df.copy()
    preprocessing.map_category_existence(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing.map_category_existence(temp_data_test)
        classification(temp_data, categories, model_tuple, args, temp_data_test)
    else:
        classification(temp_data, categories, model_tuple, args)


def classification_sentiment(reviews_df, reviews_test_df, categories, model_tuple, args):
    """
    Prepare for classification of sentiment
    :param reviews_df: train reviews
    :param reviews_test_df: test reviews
    :param categories: categories
    :param model_tuple: vector models, vector model filenames, transformation matrix and bool if model are fasttext model
    :param args: user arguments
    :return:
    """
    # positive 1, negative 0
    app_output.output("Annotated 1, not annotated 0")
    app_output.output("-----------------------\nPositive 1, negative 0\n-----------------------")
    temp_data = reviews_df.copy()
    preprocessing.map_sentiment(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing.map_sentiment(temp_data_test)
        classification(temp_data, categories, model_tuple, args, temp_data_test)
    else:
        classification(temp_data, categories, model_tuple, args)


def load_models_and_trans_matrix(args, trans_method, dict_filename):
    """
    Loads models and calculates transformation matrix
    :param args: user arguments
    :param trans_method: transformation method
    :param dict_filename: dictionary filename
    :return: vector models, vector model filenames, transformation matrix and bool if model are fasttext model
    """
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
                                                       trans_method, dict_filename)
    else:
        vec_model_test = vec_model_train

    return model_filename_train, model_filename_test, vec_model_train, vec_model_test, trans_matrix, is_fasttext


def run(args):
    """
    App controller
    :param args: user arguments
    :return:
    """
    reviews_test_df = None
    # only creating model
    if args.action == 'model':
        vector_reprezentation.create_vector_model(args)
        exit(0)
    # classification process
    elif args.action == 'cross' or args.action == 'translate' or args.action == 'monotest':
        # load test reviews
        reviews_test_df = pd.read_excel(args.reviews_path_test, sheet_name="Sheet1")
        reviews_test_df = reviews_test_df.dropna(thresh=4)

        # test reviews preprocessing
        if args.action == 'translate':
            preprocessing.translate_data(reviews_test_df, args.lang_test, args.lang)
            reviews_test_df['tokens'] = preprocessing.lower_split(reviews_test_df, args.lang)
            preprocessing.remove_bad_words(reviews_test_df['tokens'], args.lang)

        else:
            reviews_test_df['tokens'] = preprocessing.lower_split(reviews_test_df, args.lang_test)
            preprocessing.remove_bad_words(reviews_test_df['tokens'], args.lang_test)

        reviews_test_df = reviews_test_df[reviews_test_df['tokens'].apply(lambda x: x != [''])]

    # load train reviews
    reviews_df = pd.read_excel(args.reviews_path, sheet_name="Sheet1")
    reviews_df = reviews_df.dropna(thresh=4)

    # train reviews preprocessing
    reviews_df['tokens'] = preprocessing.lower_split(reviews_df, args.lang)
    preprocessing.remove_bad_words(reviews_df['tokens'], args.lang)
    reviews_df = reviews_df[reviews_df['tokens'].apply(lambda x: x != [''])]

    categories = reviews_df.columns[1:10]

    # load vector models and transformation matrix
    dict_path = constants.DICT_FOLDER + f"{args.lang}-{args.lang_test}.txt"
    model_tuple = load_models_and_trans_matrix(args, constants.DEFAULT_TRANS_METHOD, dict_path)

    # classification
    classification_category_existence(reviews_df, reviews_test_df, categories, model_tuple, args)
    classification_sentiment(reviews_df, reviews_test_df, categories, model_tuple, args)
