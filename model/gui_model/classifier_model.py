from view import app_output
from model.classifiers import lstm


class Classifier:
    """
    Object containing trained classifiers for category and sentiment detection for single category
    """
    def __init__(self, name):
        """
        Initialisation of object
        :param name: classifier method
        """
        self.name = name
        self.category_models = dict()
        self.category_models['existence'] = None
        self.category_models['sentiment'] = None

    def train_model(self, classifier_method, train_reviews, trans_matrix, target_model, device, source_lang,
                    target_lang, category_name, max_sen_len, models):
        """
        Train classifier
        :param classifier_method: classifier method
        :param train_reviews: train reviews
        :param trans_matrix: transformation matrix
        :param target_model: target vector model
        :param device: device (cpu/gpu)
        :param source_lang: source language
        :param target_lang: target language
        :param category_name: category name
        :param max_sen_len: maximal text length of data
        :param models: models in use
        :return: classifier model
        """

        X_train = train_reviews['tokens']
        Y_train = train_reviews[category_name]
        is_fasttext = True

        if classifier_method == "lstm":
            model_filename_target = models.get_model_filename(target_lang)
            model_filename_source = models.get_model_filename(source_lang)

            lstm_model = lstm.training_LSTM(target_model, trans_matrix, device, max_sen_len, X_train, Y_train,
                                            is_fasttext, batch_size=1, model_filename_train=model_filename_target,
                                            model_filename_test=model_filename_source)
            return lstm_model
        else:
            app_output.exception(f"Unknown classifier method {classifier_method}")

    def test_model(self, classifier_method, classifier_model, source_model, device, max_sen_len, words):
        """
        Test calssifier
        :param classifier_method: classifier method
        :param classifier_model: classifier model
        :param source_model: source vector model
        :param device: device (cpu/gpu)
        :param max_sen_len: maximal text length of data
        :param words: tokens to be tested
        :return: probability of positive sentiment
        """

        if classifier_method == "lstm":
            result = lstm.classifie_LSTM(classifier_model, source_model, device, max_sen_len, words)
            return result.item()

        else:
            app_output.exception(f"Unknown classifier method {classifier_method}")
