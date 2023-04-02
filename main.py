import argparse
import time

import fasttext
import pandas as pd
from gensim.models import KeyedVectors
from langdetect import detect_langs

import constants
import util
from classifires import lstm
from preprocessing import preprocessing_methods
from transformation import transformation
from models import model_methods

from nltk.corpus import stopwords


def classification_sentiments(data_df_ranked, categories, binary, args, model_tuple, test_data_df=None):
    start_time = time.time()

    model_filename_train = model_tuple[0]
    vector_filename_train = model_tuple[1]
    model_filename_test = model_tuple[2]
    vector_filename_test = model_tuple[3]
    vec_model_train = model_tuple[4]
    vec_model_test = model_tuple[5]
    trans_matrix = model_tuple[6]

    # util.print_similarity(vec_model_train, "personal")
    # print_similarity(vec_model_train, "jidlo")
    # print_similarity(vec_model_train, "cisto")
    # print_similarity(vec_model_train, "kafe")
    # print_similarity(vec_model_train, "prostredi")
    # print_similarity(vec_model_train, "drahe")
    # print_similarity(vec_model_train, "lidi")
    # print_similarity(vec_model_train, "dnes")
    # print_similarity(vec_model_train, "prachy")
    # print_similarity(vec_model_train, "zmrzlina")
    # print_similarity(vec_model_train, "piti")
    # print_similarity(vec_model_train, "rychlost")
    # print_similarity(vec_model_train, "cena")
    # print_similarity(vec_model_train, "fronta")

    # Use cuda if present
    device = util.device_recognition()

    df_test = None
    for category_name in categories:
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
                                                                                      test_size=0.20)
        else:
            if binary:
                df_test = test_data_df[test_data_df[category_name] != 2]
            else:
                df_test = test_data_df[test_data_df[category_name] != 3]
            X_train = df_sentiment['tokens']
            Y_train = df_sentiment[category_name]
            X_test = df_test['tokens']
            Y_test = df_test[category_name]

        # Plotting the sentiment distribution
        # util.plot_category_distribution(Y_train, category_name)


        util.compute_majority_class(Y_test)

        # X # LSTM with w2v/fasttext model
        max_sen_len = df_sentiment.tokens.map(len).max()
        if df_test is not None:
            max_sen_len_test = df_test.tokens.map(len).max()
            max_sen_len = max(max_sen_len, max_sen_len_test)
        lstm_model = lstm.training_LSTM(vec_model_train, trans_matrix, device, max_sen_len, X_train, Y_train, binary,
                                        batch_size=1, model_filename_train=model_filename_train,
                                        vector_filename_train=vector_filename_train,
                                        model_filename_test=model_filename_test,
                                        vector_filename_test=vector_filename_test)
        lstm.testing_LSTM(lstm_model, vec_model_test, device, max_sen_len, X_test, Y_test)

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

    true_count = 0
    false_count = 0
    with open(args.feed_path[:-4] + 'txt', 'w', encoding="utf-8") as f:
        for line in top_data_df['Text']:
            try:
                detection = [d_lang.lang == args.lang for d_lang in detect_langs(line)]
                if True in detection:
                    line = preprocessing_methods.split_line(line, args.lang)
                    f.write(line)
                    f.write('\n')
                    true_count += 1
                else:
                    false_count += 1
            except:
                false_count += 1

            # try:
            #     line = preprocessing_methods.split_line(line, args.lang)
            # except:
            #     continue
            #
            # f.write(line)
            # f.write('\n')

    util.output(f"Deleted reviews due to bad content (language, no text, ..) : {false_count}")
    util.output(f"Correct reviews : {true_count}")

    model = fasttext.train_unsupervised(args.feed_path[:-4] + 'txt', model='skipgram', dim=300)
    # filename = constants.DATA_FOLDER_UNSUPERVISED + "fasttext_unsup_" + args.lang + ".bin"
    filename = args.model_path
    model.save_model(filename)

    util.bin2vec(filename)


def create_lower_split_model(args):
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")
    # lowercase and split .,?!
    result = preprocessing_methods.lower_split(top_data_df, args.lang, check_lang=False)
    preprocessing_methods.remove_bad_words(result, args.lang)
    len_before = len(result)
    result = [x for x in result if x != ['']]
    len_after = len(result)
    print(f"Before {len_before} and after {len_after}, diff -> {len_before-len_after}")
    model_methods.make_fasttext_model(result, fasttext_file=args.model_path)


def create_token_stem_model(args):
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")

    # Tokenize the text column to get the new column 'tokenized_text'
    top_data_df = preprocessing_methods.tokenization(top_data_df, args.lang)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(top_data_df, args.lang)

    # creating model
    # models.make_word2vec_model(top_data_df, padding=False)
    # models.make_word2vec_model(top_data_df, padding=True)
    model_methods.make_fasttext_model(top_data_df['tokens'], fasttext_file=args.model_path)


def parse_agrs():
    parser = argparse.ArgumentParser(description='Reviews classification.')
    parser.add_argument('-mp', dest='model_path', type=str, help='Destination of model for classification for train'
                                                                 ' (and test for mono-lingual classification).')
    parser.add_argument('-rp', dest='reviews_path', type=str, help='Destination of reviews for classification for'
                                                                   'train (and test for mono-lingual classification).')
    parser.add_argument('-fp', dest='feed_path', type=str, help='Destination of feed for model creating.')
    parser.add_argument('-mptest', dest='model_path_test', type=str,
                        help='Destination of model for cross-lingual classification for test.')
    parser.add_argument('-mt', dest='model_type', type=str,
                        help="Type of model. 'ft' fasttext model, 'w2v' word2vec model, 'vec' vectors format.")
    parser.add_argument('-rptest', dest='reviews_path_test', type=str,
                        help='Destination of test reviews for classification for test.')
    parser.add_argument('-l', dest='lang', help='Language of train reviews (and test for mono-lingual classification).')
    parser.add_argument('-ltest', dest='lang_test', help='Language of test reviews.')
    parser.add_argument('-a', dest='action', help="Action of application. 'mono' mono-lingual classification, "
                                                  "'monotest' mono-lingual classification with separate files for "
                                                  "train and test, 'cross' cross-lingual classification, 'translate' "
                                                  "translate classification and 'model' create model to folder added "
                                                  "to 'model_filename' destination.", default='mono')

    args = parser.parse_args()
    util.output(args)

    if args.model_path is None:
        util.exception("Model path 'model_path' must be set.")
    if args.action == 'model':
        if args.feed_path is None:
            util.exception("Feed path 'feed_path' must be set.")
    else:
        if args.reviews_path is None:
            util.exception("Reviews path 'reviews_path' must be set.")
        if args.model_type is None:
            util.exception("Model type 'model_type' must be set.")
        if args.lang is None:
            util.exception("Language 'lang' must be set.")
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

    return args


def classification_sentiments_annotated(reviews_df, reviews_test_df, classes, args, model_tuple):
    # annotated 1, not annotated 0
    util.output("Annotated 1, not annotated 0")
    temp_data = reviews_df.copy()
    preprocessing_methods.map_annotated(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing_methods.map_annotated(temp_data_test)
        classification_sentiments(temp_data, classes, True, args, model_tuple, temp_data_test)
    else:
        classification_sentiments(temp_data, classes, True, args, model_tuple)


def classification_sentiments_positive(reviews_df, reviews_test_df, classes, args, model_tuple):
    # positive 1, negative 0
    util.output("Positive 1, negative 0")
    temp_data = reviews_df.copy()
    preprocessing_methods.map_sentiment(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing_methods.map_sentiment(temp_data_test)
        classification_sentiments(temp_data, classes, True, args, model_tuple, temp_data_test)
    else:
        classification_sentiments(temp_data, classes, True, args, model_tuple)


def classification_sentiments_negative(reviews_df, reviews_test_df, classes, args, model_tuple):
    # negative 1, positive 0
    util.output("Negative 1, positive 0")
    temp_data = reviews_df.copy()
    preprocessing_methods.map_sentiment_negative(temp_data)
    if reviews_test_df is not None:
        temp_data_test = reviews_test_df.copy()
        preprocessing_methods.map_sentiment_negative(temp_data_test)
        classification_sentiments(temp_data, classes, True, args, model_tuple, temp_data_test)
    else:
        classification_sentiments(temp_data, classes, True, args, model_tuple)


def load_models_and_trans_matrix(args, trans_method, filename):
    model_filename_train = None
    vector_filename_train = None
    model_filename_test = None
    vector_filename_test = None
    trans_matrix = None

    if args.model_type == 'vec':
        vector_filename_train = args.model_path
        vec_model_train = KeyedVectors.load_word2vec_format(vector_filename_train, binary=False)
        if args.action == 'cross':
            vector_filename_test = args.model_path_test
            vec_model_test = KeyedVectors.load_word2vec_format(vector_filename_test, binary=False)
            trans_matrix = transformation.get_trans_matrix(vec_model_train, vec_model_test, args.lang, args.lang_test, trans_method, filename)
        else:
            vec_model_test = vec_model_train
    else:
        model_filename_train = args.model_path
        vec_model_train = KeyedVectors.load(model_filename_train)
        vec_model_train = vec_model_train.wv
        if args.action == 'cross':
            model_filename_test = args.model_path_test
            vec_model_test = KeyedVectors.load(model_filename_test)
            vec_model_test = vec_model_test.wv
            trans_matrix = transformation.get_trans_matrix(vec_model_train, vec_model_test, args.lang, args.lang_test, trans_method, filename)
        else:
            vec_model_test = vec_model_train

    return model_filename_train, vector_filename_train, model_filename_test, vector_filename_test, vec_model_train, \
           vec_model_test, trans_matrix


def main():
    args = parse_agrs()

    reviews_test_df = None
    # only creating model
    if args.action == 'model':
        # create_token_stem_model(args)
        create_lower_split_model(args)
        # create_unsup_lower_split_model(args)
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

    # util.output("Columns in the original dataset:\n")
    # util.output(top_data_df.columns)
    # Tokenize the text column to get the new column 'tokenized_text'
    # preprocessing_methods.tokenization(reviews_df)
    # Get the stemmed_tokens
    # preprocessing_methods.stemming(reviews_df, args.lang)

    reviews_df['tokens'] = preprocessing_methods.lower_split(reviews_df, args.lang)
    preprocessing_methods.remove_bad_words(reviews_df['tokens'], args.lang)
    reviews_df = reviews_df[reviews_df['tokens'].apply(lambda x: x != [''])]

    classes = reviews_df.columns[1:10]

    trans_method, filename = "orto1", constants.DICT_FOLDER + f"{args.lang}-{args.lang_test}_muj.txt"
    model_tuple = load_models_and_trans_matrix(args, trans_method, filename)

    classification_sentiments_annotated(reviews_df, reviews_test_df, classes, args, model_tuple)
    classification_sentiments_positive(reviews_df, reviews_test_df, classes, args, model_tuple)

if __name__ == "__main__":
    main()
