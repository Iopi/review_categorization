import re

import constants
from model.enums.language_enum import Language
from view import app_output
from easynmt import EasyNMT
from langdetect import detect_langs
from sklearn.model_selection import train_test_split


def remove_bad_words(sentences, lang):
    """
    Removing the stop words from sentences
    :param sentences: sentences
    :param lang: language
    :return:
    """
    if lang == Language.CZECH.value:
        filename = constants.SW_FOLDER + "stop_words_czech.txt"
    elif lang == Language.ENGLISH.value:
        filename = constants.SW_FOLDER + "stop_words_english.txt"
    elif lang == Language.GERMAN.value:
        filename = constants.SW_FOLDER + "stop_words_german.txt"
    else:
        app_output.exception(f"Unknown language {lang}.")

    with open(filename, encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]

    for sentence in sentences:
        sentence[:] = list(filter(lambda word: word not in stop_words, sentence))


def get_rank_id_existence(sentiment):
    """
    Get rank by existence of sentiment
    :param sentiment: sentiment
    :return: 1 if sentiment exists, 0 if not
    """
    if sentiment == "Positive" or sentiment == "Negative" or sentiment == "Neutral":
        return 1
    else:
        return 0


def get_rank_id_positive(sentiment):
    """
    Get rank by sentiment
    :param sentiment: sentiment
    :return: 1 if sentiment is Positive, 0 if Negative
    """
    if sentiment == "Positive":
        return 1
    elif sentiment == "Negative":
        return 0
    else:
        return 2


def map_category_existence(data_df):
    """
    Maps rank id by existence of sentiment for each category
    :param data_df: data dataframe
    :return:
    """
    data_df['General'] = [get_rank_id_existence(x) for x in data_df['General']]
    data_df['Food'] = [get_rank_id_existence(x) for x in data_df['Food']]
    data_df['Drink'] = [get_rank_id_existence(x) for x in data_df['Drink']]
    data_df['Staff'] = [get_rank_id_existence(x) for x in data_df['Staff']]
    data_df['Speed'] = [get_rank_id_existence(x) for x in data_df['Speed']]
    data_df['Cleanness'] = [get_rank_id_existence(x) for x in data_df['Cleanness']]
    data_df['Prices'] = [get_rank_id_existence(x) for x in data_df['Prices']]
    data_df['Environment'] = [get_rank_id_existence(x) for x in data_df['Environment']]
    data_df['Occupancy'] = [get_rank_id_existence(x) for x in data_df['Occupancy']]
    data_df['Other'] = [get_rank_id_existence(x) for x in data_df['Other']]


def map_sentiment(data_df):
    """
    Maps rank id by sentiment for each category
    :param data_df: data dataframe
    :return:
    """
    data_df['General'] = [get_rank_id_positive(x) for x in data_df['General']]
    data_df['Food'] = [get_rank_id_positive(x) for x in data_df['Food']]
    data_df['Drink'] = [get_rank_id_positive(x) for x in data_df['Drink']]
    data_df['Staff'] = [get_rank_id_positive(x) for x in data_df['Staff']]
    data_df['Speed'] = [get_rank_id_positive(x) for x in data_df['Speed']]
    data_df['Cleanness'] = [get_rank_id_positive(x) for x in data_df['Cleanness']]
    data_df['Prices'] = [get_rank_id_positive(x) for x in data_df['Prices']]
    data_df['Environment'] = [get_rank_id_positive(x) for x in data_df['Environment']]
    data_df['Occupancy'] = [get_rank_id_positive(x) for x in data_df['Occupancy']]
    data_df['Other'] = [get_rank_id_positive(x) for x in data_df['Other']]


def split_train_test(data_df, category_name, test_size=0.3, shuffle_state=True):
    """
    Splits the data for train and test part
    :param data_df: data dataframe
    :param category_name: category name
    :param test_size: test size
    :param shuffle_state: shuffle state
    :return: split data
    """
    X_train, X_test, Y_train, Y_test = train_test_split(data_df['tokens'],
                                                        data_df[category_name],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)

    X_train = X_train.reset_index()['tokens']
    X_test = X_test.reset_index()['tokens']
    Y_train = Y_train.reset_index()[category_name]
    Y_test = Y_test.reset_index()[category_name]
    return X_train, X_test, Y_train, Y_test


def translate_data(data_df, lang_from, lang_to):
    """
    Translates data
    :param data_df: data dataframe
    :param lang_from: source language
    :param lang_to: target language
    :return:
    """
    model = EasyNMT('opus-mt')
    data_df['Text'] = [model.translate(x, target_lang=lang_to, source_lang=lang_from) for x in data_df['Text']]


def lower_split(data_df, lang, check_lang=False):
    """
    Data preprocessing - lowercasing, tokenization, remove symbols, symbols offset, remove diacritics
    :param data_df: data dataframe
    :param lang: language of data
    :param check_lang: if true use only text lines in correct language
    :return: preprocessed data
    """
    lost = 0
    success = 0
    result = []
    # if true use only text lines in correct language
    if check_lang:
        for line in data_df['Text']:
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
        for line in data_df['Text']:
            try:
                line = line.lower()
                line = split_line(line, lang)
                result.append(line.split(" "))
                success += 1
            except:
                lost += 1

    # app_output.output(f"Deleted reviews due to bad content (language, no text, ..) : {lost}")
    app_output.output(f"Correct reviews for language {lang} : {success}")
    return result


def split_line(line, lang):
    """
    Tokenization, remove symbols, symbols offset, remove diacritics
    :param line: text line
    :param lang: language of line
    :return: preprocessed line
    """
    # remove diacritics
    if lang == Language.CZECH.value:
        line = remove_diacritics_cs(line)
    elif lang == Language.GERMAN.value:
        line = remove_diacritics_de(line)
    line = re.sub('([.,!?()])', r' \1 ', line)  # punctuation offset
    line = re.sub(r'[’\'\-]', '', line)  # remove unwanted symbols ', -
    line = re.sub(r'[":;]', ' ', line)  # replace some unwanted symbols with space
    line = re.sub('\s{2,}', ' ', line)  # united spaces

    return line.lstrip().rstrip()


def remove_diacritics_de(sentence):
    """
    Removes diacritics for German
    :param sentence: sentence
    :return: sentence without diacritics
    """
    old_symbols = "äöü"
    new_symbols = "aou"

    sentence = sentence.replace("ß", "ss")

    for i in range(len(old_symbols)):
        sentence = sentence.replace(old_symbols[i], new_symbols[i])

    return sentence


def remove_diacritics_cs(sentence):
    """
    Removes diacritics for Czech
    :param sentence: sentence
    :return: sentence without diacritics
    """
    old_symbols = "áčďéěíňóřšťúůýž"
    new_symbols = "acdeeinorstuuyz"

    for i in range(len(old_symbols)):
        sentence = sentence.replace(old_symbols[i], new_symbols[i])

    return sentence


def remove_diacritics_from_file(filename, langs):
    """
    Removes diacritics from file
    :param filename: filename
    :param langs: language of text in file
    :return:
    """
    with open(filename, 'r', encoding="utf-8") as f:
        for lang in langs:
            if lang == Language.CZECH.value:
                lines = [remove_diacritics_cs(line) for line in f]
            elif lang == Language.GERMAN.value:
                lines = [remove_diacritics_de(line) for line in f]

    with open(filename, 'w', encoding="utf-8") as f:
        f.writelines(lines)

    app_output.output(f"Removed diacritics from file {filename}")
