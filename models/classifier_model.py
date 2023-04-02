import constants
import util
from classifires import lstm
from models import saved_model


class Classifier:
    def __init__(self, name):
        self.name = name
        self.categories = dict()
        self.prepare_categories()

    def prepare_categories(self):
        for category in constants.CATEGORIES:
            self.categories[category] = dict()
            self.categories[category]['existence'] = None
            self.categories[category]['sentiment'] = None

    def train_model(self, classifier_method, train_reviews, trans_matrix, target_model, device, target_lang,
                    source_lang, category_name, max_sen_len, models):

        X_train = train_reviews['tokens']
        Y_train = train_reviews[category_name]

        if classifier_method == "lstm":
            model_filename_target = models.get_model_filename(target_lang)
            model_filename_source = models.get_model_filename(source_lang)

            lstm_model = lstm.training_LSTM(target_model, trans_matrix, device, max_sen_len, X_train, Y_train, True,
                                            batch_size=1, model_filename_train=model_filename_target,
                                            vector_filename_train=None,
                                            model_filename_test=model_filename_source,
                                            vector_filename_test=None)
            return lstm_model
        else:
            util.exception(f"Unknown classifier method {classifier_method}")

    def test_model(self, classifier_method, classifier_model, source_model, device, max_sen_len, words):

        if classifier_method == "lstm":
            result = lstm.classifie_LSTM(classifier_model, source_model, device, max_sen_len, words)
            return result.item()

        else:
            util.exception(f"Unknown classifier method {classifier_method}")

