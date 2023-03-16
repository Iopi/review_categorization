import os
import time
import numpy as np
import pandas as pd
import torch
from googletrans import Translator
from easynmt import EasyNMT

from sklearn.model_selection import train_test_split
import re

import constants
import util
from preprocessing import czech_stemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from langdetect import detect, detect_langs

from vector_model import models


def remove_bad_words(sentences, lang):
    """
    Removing the stop words
    """
    if lang == 'cs':
        filename = constants.SW_FOLDER + "stop_words_czech.txt"
    elif lang == 'en':
        filename = constants.SW_FOLDER + "stop_words_english.txt"
    elif lang == 'de':
        filename = constants.SW_FOLDER + "stop_words_german.txt"
    else:
        raise Exception("Unknown language.")

    with open(filename, encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]

    for sentence in sentences:
        sentence[:] = list(filter(lambda word: word not in stop_words, sentence))

    # sentences[:] = [sentence for sentence in sentences if sentence != ['']]


def tokenization(top_data_df, lang=None):
    """
    Tokenize the text column to get the new column 'tokenized_text'
    :param lang:
    :param top_data_df:
    :return:
    """

    tokens = []
    if lang is not None:
        correct_row = []

        for line in top_data_df['Text']:
            try:
                detection = [d_lang.lang == lang for d_lang in detect_langs(line)]
                if True in detection:
                    tokens.append(simple_preprocess(line, deacc=True))
                    correct_row.append(True)
                else:
                    correct_row.append(False)
            except:
                correct_row.append(False)

        values, counts = np.unique(correct_row, return_counts=True)

        false_count = 0
        if len(values) != 2:
            if values[0] == False:
                raise Exception("All reviews have bad content.")
            true_count = counts[0]
        else:
            if values[0] == False:
                false_count = counts[0]
                true_count = counts[1]
            else:
                false_count = counts[1]
                true_count = counts[0]

        util.output(f"Deleted reviews due to bad content (language, no text, ..) : {false_count}")
        util.output(f"Correct reviews : {true_count}")
        top_data_df['temp_correct'] = correct_row
        top_data_df = top_data_df[top_data_df.temp_correct]
        top_data_df.drop(columns='temp_correct')
    else:
        for line in top_data_df['Text']:
            tokens.append(simple_preprocess(line, deacc=True))

    top_data_df['tokenized_text'] = tokens
    # top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['Text'].head(10)]
    # util.output(top_data_df['tokenized_text'].head(10))

    return top_data_df


def stemming(top_data_df_small, lang):
    if lang == 'cs':

        # tagger = Tagger("D:\skola\Diplomka\data\morpho\czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger")
        # tokens = list(tagger.tag(top_data_df_small['tokens'], convert ='strip_lemma_id'))
        # top_data_df_small['tokens'] = tokens
        top_data_df_small['tokens'] = [[czech_stemmer.cz_stem(word) for word in tokens] for tokens in
                                       top_data_df_small['tokenized_text']]
        # util.output(top_data_df_small['tokens'].head(10))

    elif lang == 'en':
        porter = PorterStemmer()
        top_data_df_small['tokens'] = [[porter.stem(word) for word in tokens] for tokens in
                                       top_data_df_small['tokenized_text']]


def get_rank_id_annotated(sentiment):
    if sentiment == "Positive" or sentiment == "Negative" or sentiment == "Neutral":
        return 1
    else:
        return 0


def get_rank_id_positive(sentiment):
    if sentiment == "Positive":
        return 1
    elif sentiment == "Negative": # or sentiment == "Neutral":
        return 0
    else:
        return 2


def get_rank_id_negative(sentiment):
    if sentiment == "Negative":
        return 1
    elif sentiment == "Positive": # or sentiment == "Neutral":
        return 0
    else:
        return 2


def get_rank_id(sentiment):
    if sentiment == "Neutral":
        return 0
    elif sentiment == "Positive":
        return 1
    elif sentiment == "Negative":
        return 2
    else:
        return 3


def map_sentiment_annotated(top_data_df):
    top_data_df['General'] = [get_rank_id_annotated(x) for x in top_data_df['General']]
    top_data_df['Food'] = [get_rank_id_annotated(x) for x in top_data_df['Food']]
    top_data_df['Drink'] = [get_rank_id_annotated(x) for x in top_data_df['Drink']]
    top_data_df['Staff'] = [get_rank_id_annotated(x) for x in top_data_df['Staff']]
    top_data_df['Speed'] = [get_rank_id_annotated(x) for x in top_data_df['Speed']]
    top_data_df['Cleanness'] = [get_rank_id_annotated(x) for x in top_data_df['Cleanness']]
    top_data_df['Prices'] = [get_rank_id_annotated(x) for x in top_data_df['Prices']]
    top_data_df['Environment'] = [get_rank_id_annotated(x) for x in top_data_df['Environment']]
    top_data_df['Occupancy'] = [get_rank_id_annotated(x) for x in top_data_df['Occupancy']]
    top_data_df['Other'] = [get_rank_id_annotated(x) for x in top_data_df['Other']]


def map_sentiment_positive(top_data_df):
    top_data_df['General'] = [get_rank_id_positive(x) for x in top_data_df['General']]
    top_data_df['Food'] = [get_rank_id_positive(x) for x in top_data_df['Food']]
    top_data_df['Drink'] = [get_rank_id_positive(x) for x in top_data_df['Drink']]
    top_data_df['Staff'] = [get_rank_id_positive(x) for x in top_data_df['Staff']]
    top_data_df['Speed'] = [get_rank_id_positive(x) for x in top_data_df['Speed']]
    top_data_df['Cleanness'] = [get_rank_id_positive(x) for x in top_data_df['Cleanness']]
    top_data_df['Prices'] = [get_rank_id_positive(x) for x in top_data_df['Prices']]
    top_data_df['Environment'] = [get_rank_id_positive(x) for x in top_data_df['Environment']]
    top_data_df['Occupancy'] = [get_rank_id_positive(x) for x in top_data_df['Occupancy']]
    top_data_df['Other'] = [get_rank_id_positive(x) for x in top_data_df['Other']]


def map_sentiment_negative(top_data_df):
    top_data_df['General'] = [get_rank_id_negative(x) for x in top_data_df['General']]
    top_data_df['Food'] = [get_rank_id_negative(x) for x in top_data_df['Food']]
    top_data_df['Drink'] = [get_rank_id_negative(x) for x in top_data_df['Drink']]
    top_data_df['Staff'] = [get_rank_id_negative(x) for x in top_data_df['Staff']]
    top_data_df['Speed'] = [get_rank_id_negative(x) for x in top_data_df['Speed']]
    top_data_df['Cleanness'] = [get_rank_id_negative(x) for x in top_data_df['Cleanness']]
    top_data_df['Prices'] = [get_rank_id_negative(x) for x in top_data_df['Prices']]
    top_data_df['Environment'] = [get_rank_id_negative(x) for x in top_data_df['Environment']]
    top_data_df['Occupancy'] = [get_rank_id_negative(x) for x in top_data_df['Occupancy']]
    top_data_df['Other'] = [get_rank_id_negative(x) for x in top_data_df['Other']]


def map_sentiment(top_data_df):
    top_data_df['General'] = [get_rank_id(x) for x in top_data_df['General']]
    top_data_df['Food'] = [get_rank_id(x) for x in top_data_df['Food']]
    top_data_df['Drink'] = [get_rank_id(x) for x in top_data_df['Drink']]
    top_data_df['Staff'] = [get_rank_id(x) for x in top_data_df['Staff']]
    top_data_df['Speed'] = [get_rank_id(x) for x in top_data_df['Speed']]
    top_data_df['Cleanness'] = [get_rank_id(x) for x in top_data_df['Cleanness']]
    top_data_df['Prices'] = [get_rank_id(x) for x in top_data_df['Prices']]
    top_data_df['Environment'] = [get_rank_id(x) for x in top_data_df['Environment']]
    top_data_df['Occupancy'] = [get_rank_id(x) for x in top_data_df['Occupancy']]
    top_data_df['Other'] = [get_rank_id(x) for x in top_data_df['Other']]


def split_train_test(top_data_df_small, class_name, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small['tokens'],
                                                        top_data_df_small[class_name],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    # util.output("Value counts for Train sentiments")
    # util.output(Y_train.value_counts())
    # util.output("Value counts for Test sentiments")
    # util.output(Y_test.value_counts())

    X_train = X_train.reset_index()['tokens']
    X_test = X_test.reset_index()['tokens']
    Y_train = Y_train.reset_index()[class_name]
    Y_test = Y_test.reset_index()[class_name]
    # util.output(X_train.head())
    return X_train, X_test, Y_train, Y_test


# Function to get the util.output tensor
def make_target(label, device):
    if label == -1:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 0:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)


def translate_data(data_df, lang_from, lang_to):
    model = EasyNMT('opus-mt')
    data_df['Text'] = [model.translate(x, target_lang=lang_to, source_lang=lang_from) for x in data_df['Text']]
    # print(data_df['Text'].head())


def get_translate_text(path, lang_from, lang_to):
    data_df = pd.read_excel(path, sheet_name="Sheet1")
    data_df = data_df.dropna(thresh=4)
    if lang_from != lang_to:
        translate_data(data_df, lang_from, lang_to)
    tokenization(data_df)
    stemming(data_df, lang_to)

    return data_df


def lower_split(top_data_df, lang, check_lang=False):

    # top_data_df['lower_split'] = top_data_df['Text'].str.lower()
    lost = 0
    success = 0
    result = []
    if check_lang:
        for line in top_data_df['Text']:
            try:
                detection = [d_lang.lang == lang for d_lang in detect_langs(line)]
                if True in detection:
                    line = line.lower()
                    line = split_line(line, lang)
                    result.append(line.split(" "))
                    success += 1
                else:
                    lost += 1
            except:
                lost += 1
    else:
        for line in top_data_df['Text']:
            try:
                line = line.lower()
                line = split_line(line, lang)
                result.append(line.split(" "))
                success += 1
            except:
                lost += 1

    util.output(f"Deleted reviews due to bad content (language, no text, ..) : {lost}")
    util.output(f"Correct reviews : {success}")
    # top_data_df.drop('lower_split', axis=1, inplace=True)
    return result


def split_line(line, lang):
    if lang == 'cs':
        line = remove_diacritics_cs(line)
    elif lang == 'de':
        line = remove_diacritics_de(line)
        pass
    # line = re.sub('([.,!?()])', r' \1 ', line) # odsazeni punktace
    line = re.sub(r'[.,"\'-?:!;]', ' ', line)  # smazani punktace a nechtenych symbolu
    line = re.sub('\s{2,}', ' ', line)  # sjednoceni mezer

    return line.rstrip()

def remove_diacritics_de(sentence):
    old_symbols = "äöü"
    new_symbols = "aou"

    sentence = sentence.replace("ß", "ss")

    for i in range(len(old_symbols)):
        sentence = sentence.replace(old_symbols[i], new_symbols[i])

    return sentence

def remove_diacritics_cs(sentence):
    old_symbols = "áčďéěíňóřšťúůýž"
    new_symbols = "acdeeinorstuuyz"

    for i in range(len(old_symbols)):
        sentence = sentence.replace(old_symbols[i], new_symbols[i])

    return sentence


