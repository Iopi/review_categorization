import util
from model.classifires import lstm


class Classifier:
    def __init__(self, name):
        self.name = name
        self.category_models = dict()
        self.category_models['existence'] = None
        self.category_models['sentiment'] = None

    def train_model(self, classifier_method, train_reviews, trans_matrix, target_model, device, source_lang,
                    target_lang, category_name, max_sen_len, models):

        X_train = train_reviews['tokens']
        Y_train = train_reviews[category_name]
        is_fasttext = True

        if classifier_method == "lstm":
            model_filename_target = models.get_model_filename(target_lang)
            model_filename_source = models.get_model_filename(source_lang)

            lstm_model = lstm.training_LSTM(target_model, trans_matrix, device, max_sen_len, X_train, Y_train, True,
                                            is_fasttext, batch_size=1, model_filename_train=model_filename_target,
                                            model_filename_test=model_filename_source)
            return lstm_model
        else:
            util.exception(f"Unknown classifier method {classifier_method}")

    def test_model(self, classifier_method, classifier_model, source_model, device, max_sen_len, words):

        if classifier_method == "lstm":
            result = lstm.classifie_LSTM(classifier_model, source_model, device, max_sen_len, words)
            return result.item()

        else:
            util.exception(f"Unknown classifier method {classifier_method}")
