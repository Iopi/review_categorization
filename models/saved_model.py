import constants
import pandas as pd
import util
from enums.language_enum import Language
from gensim.models import KeyedVectors
from preprocessing import preprocessing_methods
from transformation import transformation


class SavedModels:
    def __init__(self):
        self.czech_model = None
        self.english_model = None
        self.german_model = None
        self.czech_model_filename = None
        self.english_model_filename = None
        self.german_model_filename = None
        self.trans_matrix_cs_en = None
        self.trans_matrix_cs_de = None
        self.reviews_cs = None
        self.reviews_en = None
        self.reviews_de = None

    def prepare_vec_model(self, lang):
        if lang == Language.CZECH.value:
            if self.czech_model is None:
                self.czech_model = KeyedVectors.load(constants.DEFAULT_VEC_MODEL_CS)
                self.czech_model = self.czech_model.wv
                self.czech_model_filename = constants.DEFAULT_VEC_MODEL_CS
            return self.czech_model

        elif lang == Language.ENGLISH.value:
            if self.english_model is None:
                self.english_model = KeyedVectors.load(constants.DEFAULT_VEC_MODEL_EN)
                self.english_model = self.english_model.wv
                self.english_model_filename = constants.DEFAULT_VEC_MODEL_EN
            return self.english_model

        elif lang == Language.GERMAN.value:
            if self.german_model is None:
                self.german_model = KeyedVectors.load(constants.DEFAULT_VEC_MODEL_DE)
                self.german_model = self.german_model.wv
                self.german_model_filename = constants.DEFAULT_VEC_MODEL_DE
            return self.german_model

        else:
            util.exception(f"Unknown language {lang}")

    def compute_transform_matrix(self, trans_method, target_lang, source_lang):
        dict_filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}_muj.txt"

        if Language.CZECH.value == target_lang:
            if self.czech_model is None:
                util.exception(f"Vector model for czech language not exists.")
            if Language.ENGLISH.value == source_lang:
                if self.english_model is None:
                    util.exception(f"Vector model for english language not exists.")
                if self.trans_matrix_cs_en is None:
                    self.trans_matrix_cs_en = transformation.get_trans_matrix(self.czech_model, self.english_model,
                                                                              target_lang, source_lang, trans_method,
                                                                              dict_filename)
                return self.trans_matrix_cs_en
            elif Language.GERMAN.value == source_lang:
                if self.german_model is None:
                    util.exception(f"Vector model for german language not exists.")
                if self.trans_matrix_cs_de is None:
                    self.trans_matrix_cs_de = transformation.get_trans_matrix(self.czech_model, self.german_model,
                                                                              target_lang, source_lang, trans_method,
                                                                              dict_filename)
                return self.trans_matrix_cs_de

        util.exception(
            f"Not found transformation method for target language {target_lang} and source language {source_lang}.")

    def load_reviews(self, lang):
        if lang == Language.CZECH.value:
            if self.reviews_cs is None:
                self.reviews_cs = pd.read_excel(constants.DEFAULT_REVIEWS_CS, sheet_name="Sheet1")
                self.reviews_cs = self.reviews_cs.dropna(thresh=4)
                self.reviews_cs['tokens'] = preprocessing_methods.lower_split(self.reviews_cs, lang)
                preprocessing_methods.remove_bad_words(self.reviews_cs['tokens'], lang)
                self.reviews_cs = self.reviews_cs[self.reviews_cs['tokens'].apply(lambda x: x != [''])]
            return self.reviews_cs

        if lang == Language.ENGLISH.value:
            if self.reviews_en is None:
                self.reviews_en = pd.read_excel(constants.DEFAULT_REVIEWS_EN, sheet_name="Sheet1")
                self.reviews_en = self.reviews_en.dropna(thresh=4)
                self.reviews_en['tokens'] = preprocessing_methods.lower_split(self.reviews_en, lang)
                preprocessing_methods.remove_bad_words(self.reviews_en['tokens'], lang)
                self.reviews_en = self.reviews_en[self.reviews_en['tokens'].apply(lambda x: x != [''])]
            return self.reviews_en

        if lang == Language.GERMAN.value:
            if self.reviews_de is None:
                self.reviews_de = pd.read_excel(constants.DEFAULT_REVIEWS_DE, sheet_name="Sheet1")
                self.reviews_de = self.reviews_de.dropna(thresh=4)
                self.reviews_de['tokens'] = preprocessing_methods.lower_split(self.reviews_de, lang)
                preprocessing_methods.remove_bad_words(self.reviews_de['tokens'], lang)
                self.reviews_de = self.reviews_de[self.reviews_de['tokens'].apply(lambda x: x != [''])]
            return self.reviews_de

    def get_model_filename(self, lang):
        if lang == Language.CZECH.value:
            if self.czech_model_filename is not None:
                return self.czech_model_filename

        if lang == Language.ENGLISH.value:
            if self.english_model_filename is not None:
                return self.english_model_filename

        if lang == Language.GERMAN.value:
            if self.german_model_filename is not None:
                return self.german_model_filename

        util.exception(f"Model filename for language {lang} not found.")
