DATA_FOLDER = 'data/'
SW_FOLDER = DATA_FOLDER + 'stopwords/'
DICT_FOLDER = DATA_FOLDER + 'dictionary/'
MODEL_FOLDER = DATA_FOLDER + 'vec_model/'
CLASSIFIER_FOLDER = DATA_FOLDER + 'saved_classifiers/'


CATEGORIES = ['General', 'Food', 'Drink', 'Staff', 'Speed', 'Cleanness', 'Prices', 'Environment', 'Occupancy']

# GUI defaults
DEFAULT_TRANS_METHOD = 'orto'
DEFAULT_VEC_MODEL_CS = DATA_FOLDER + 'vec_model/ft_cs.bin'
DEFAULT_VEC_MODEL_EN = DATA_FOLDER + 'vec_model/ft_en.bin'
DEFAULT_VEC_MODEL_DE = DATA_FOLDER + 'vec_model/ft_de.bin'

DEFAULT_REVIEWS_CS = DATA_FOLDER + 'review/reviews_cs.xlsx'
DEFAULT_REVIEWS_EN = DATA_FOLDER + 'review/reviews_en.xlsx'
DEFAULT_REVIEWS_DE = DATA_FOLDER + 'review/reviews_de.xlsx'

CLASSIFIER_LSTM = DATA_FOLDER + 'saved_classifiers/lstm_models_'
