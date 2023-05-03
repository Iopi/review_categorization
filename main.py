import argparse
from view import app_output
from controller import app_controller


def parse_agrs():
    '''
    Arguments parser
    '''
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

    if args.model_type is None:
        app_output.exception("Model type 'model_type' must be set.")
    if args.model_type == 'w2v' or args.model_type == 'ft':
        if args.model_path is None:
            app_output.exception("Model path 'model_path' must be set.")
    if args.lang is None:
        app_output.exception("Language 'lang' must be set.")
    if args.action == 'model':
        if args.feed_path is None:
            app_output.exception("Feed path 'feed_path' must be set.")
    else:
        if args.classi_model is None:
            app_output.exception("Classification model 'classi_model' must be set.")
        if args.reviews_path is None:
            app_output.exception("Reviews path 'reviews_path' must be set.")
        if args.action == 'cross':
            if args.model_path_test is None:
                app_output.exception("Model path for test 'model_path_test' must be set.")
            if args.reviews_path_test is None:
                app_output.exception("Reviews path for test 'reviews_path_test' must be set.")
            if args.lang_test is None:
                app_output.exception("Language for test 'lang_test' must be set.")
        elif args.action == 'monotest' or args.action == 'translate':
            if args.reviews_path_test is None:
                app_output.exception("Reviews path for test 'reviews_path_test' must be set.")
            if args.lang_test is None:
                app_output.exception("Language for test 'lang_test' must be set.")
        elif args.action != 'mono':
            app_output.exception(
                "Wrong argument for action. Action of application. 'mono' mono-lingual classification, 'cross' "
                "cross-lingual classification and 'model' create model to folder "
                "added to 'model_filename' destination.")

        if ((args.model_type == 'lstm' or args.model_type == 'cnn') and (args.classi_model == 'bow' or
                                                                         args.classi_model == 'tfidf')) or (
                (args.model_type == 'svm' or args.model_type == 'logreg' or
                 args.model_type == 'dectree') and (args.classi_model == 'w2v' or
                                                    args.classi_model == 'ft')):
            app_output.exception("Neural network classifiers (lstm, cnn) can only be in combination with text representation "
                           "word2vec and fasttext and other classifiers (Support vector machines, Logistic regression "
                           "and Decision tree) can only be in combination with bow and tf-idf text representation")
    return args


def main():
    '''
    Main runs control af arguments and app controller
    '''
    args = parse_agrs()

    app_controller.run(args)


if __name__ == "__main__":
    main()
