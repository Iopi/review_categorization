import time
import pandas as pd
from gensim.models import KeyedVectors

import constants
from controller import preprocessing, transformation, vector_reprezentation
from model.classifiers import svm, lstm, cnn, log_reg, decision_tree
from view import app_output
import util


def classification_sentiments(data_df_ranked, categories, model_tuple, args, test_data_df=None):
    '''
    Detection of sentiment for all categories
    :param data_df_ranked: Main window of ui
    :param categories: Main window of ui
    :param model_tuple: Main window of ui
    :param args: Main window of ui
    :param test_data_df: Main window of ui
    '''

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

        app_output.output("-------------------------------------")
        app_output.output("Classification sentiment " + category_name)

        # drop not needed rows
        df_sentiment = data_df_ranked[data_df_ranked[category_name] != 2]

        # Plotting the sentiment distribution
        # util.plot_distribution(df_sentiment, category_name)

        # util.sentiment_count(df_sentiment, category_name)

        # Call the train_test_split
        if test_data_df is None:
            X_train, X_test, Y_train, Y_test = preprocessing.split_train_test(df_sentiment, category_name,
                                                                              test_size=0.20)
        else:
            df_test = test_data_df[test_data_df[category_name] != 2]

            # util.sentiment_count(df_test, category_name)

            X_train = df_sentiment['tokens']
            Y_train = df_sentiment[category_name]
            X_test = df_test['tokens']
            Y_test = df_test[category_name]

        util.compute_majority_class(Y_test)
        app_output.output("-------------------------------------")

        # if simple classifier
        if vec_model_train is None:
            clf = None
            filename = None
            tfidf_model = None
            review_dict = vector_reprezentation.make_dict(df_sentiment, padding=False)
            # create text representation
            # bag of words vector representation
            if args.model_type == 'bow':
                filename = constants.MODEL_FOLDER + 'train_review_bow.csv'
                vector_reprezentation.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
                clf = decision_tree.training_Decision_Tree(Y_train, filename)

            # tf-idf vector representation
            elif args.model_type == 'tfidf':
                filename = constants.MODEL_FOLDER + 'train_review_tfidf.csv'
                tfidf_model = vector_reprezentation.create_tfidf_model_file(review_dict, df_sentiment, X_train,
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
                vector_reprezentation.testing_classificator_with_bow(clf, review_dict, X_test, Y_test)

            elif args.model_type == 'tfidf':
                vector_reprezentation.testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test)

        # if neural network
        else:
            max_sen_len = df_sentiment.tokens.map(len).max()
            if df_test is not None:
                max_sen_len_test = df_test.tokens.map(len).max()
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


def classification_category_existence(reviews_df, reviews_test_df, classes, model_tuple, args):

    # annotated 1, not annotated 0
    app_output.output("-----------------------------\nAnnotated 1, not annotated 0\n-----------------------------")
    temp_data = reviews_df.copy()
    preprocessing.map_category_existence(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing.map_category_existence(temp_data_test)
        classification_sentiments(temp_data, classes, model_tuple, args, temp_data_test)
    else:
        classification_sentiments(temp_data, classes, model_tuple, args)


def classification_sentiment(reviews_df, reviews_test_df, classes, model_tuple, args):
    # positive 1, negative 0
    app_output.output("Annotated 1, not annotated 0")
    app_output.output("-----------------------\nPositive 1, negative 0\n-----------------------")
    temp_data = reviews_df.copy()
    preprocessing.map_sentiment(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing.map_sentiment(temp_data_test)
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


def run(args):
    reviews_test_df = None
    # only creating model
    if args.action == 'model':
        vector_reprezentation.create_lower_split_model(args)
        exit(0)
    # classification
    elif args.action == 'cross' or args.action == 'translate' or args.action == 'monotest':
        reviews_test_df = pd.read_excel(args.reviews_path_test, sheet_name="Sheet1")
        reviews_test_df = reviews_test_df.dropna(thresh=4)
        if args.action == 'translate':
            preprocessing.translate_data(reviews_test_df, args.lang_test, args.lang)
            reviews_test_df['tokens'] = preprocessing.lower_split(reviews_test_df, args.lang)
            preprocessing.remove_bad_words(reviews_test_df['tokens'], args.lang)

        else:
            reviews_test_df['tokens'] = preprocessing.lower_split(reviews_test_df, args.lang_test)
            preprocessing.remove_bad_words(reviews_test_df['tokens'], args.lang_test)

        reviews_test_df = reviews_test_df[reviews_test_df['tokens'].apply(lambda x: x != [''])]

    reviews_df = pd.read_excel(args.reviews_path, sheet_name="Sheet1")
    reviews_df = reviews_df.dropna(thresh=4)

    reviews_df['tokens'] = preprocessing.lower_split(reviews_df, args.lang)
    preprocessing.remove_bad_words(reviews_df['tokens'], args.lang)
    reviews_df = reviews_df[reviews_df['tokens'].apply(lambda x: x != [''])]

    classes = reviews_df.columns[1:10]

    dict_path = constants.DICT_FOLDER + f"{args.lang}-{args.lang_test}.txt"
    model_tuple = load_models_and_trans_matrix(args, constants.DEFAULT_TRANS_METHOD, dict_path)

    classification_category_existence(reviews_df, reviews_test_df, classes, model_tuple, args)
    classification_sentiment(reviews_df, reviews_test_df, classes, model_tuple, args)
