import re

import constants
from model.enums.language_enum import Language
from view import app_output
from easynmt import EasyNMT
from langdetect import detect_langs
from sklearn.model_selection import train_test_split


def remove_bad_words(sentences, lang):
    """
    Removing the stop words
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
    if sentiment == "Positive" or sentiment == "Negative" or sentiment == "Neutral":
        return 1
    else:
        return 0


def get_rank_id_positive(sentiment):
    if sentiment == "Positive":
        return 1
    elif sentiment == "Negative":  # or sentiment == "Neutral":
        return 0
    else:
        return 2


def map_category_existence(top_data_df):
    top_data_df['General'] = [get_rank_id_existence(x) for x in top_data_df['General']]
    top_data_df['Food'] = [get_rank_id_existence(x) for x in top_data_df['Food']]
    top_data_df['Drink'] = [get_rank_id_existence(x) for x in top_data_df['Drink']]
    top_data_df['Staff'] = [get_rank_id_existence(x) for x in top_data_df['Staff']]
    top_data_df['Speed'] = [get_rank_id_existence(x) for x in top_data_df['Speed']]
    top_data_df['Cleanness'] = [get_rank_id_existence(x) for x in top_data_df['Cleanness']]
    top_data_df['Prices'] = [get_rank_id_existence(x) for x in top_data_df['Prices']]
    top_data_df['Environment'] = [get_rank_id_existence(x) for x in top_data_df['Environment']]
    top_data_df['Occupancy'] = [get_rank_id_existence(x) for x in top_data_df['Occupancy']]
    top_data_df['Other'] = [get_rank_id_existence(x) for x in top_data_df['Other']]


def map_sentiment(top_data_df):
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


def split_train_test(top_data_df_small, class_name, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small['tokens'],
                                                        top_data_df_small[class_name],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)

    X_train = X_train.reset_index()['tokens']
    X_test = X_test.reset_index()['tokens']
    Y_train = Y_train.reset_index()[class_name]
    Y_test = Y_test.reset_index()[class_name]
    return X_train, X_test, Y_train, Y_test


def translate_data(data_df, lang_from, lang_to):
    model = EasyNMT('opus-mt')
    data_df['Text'] = [model.translate(x, target_lang=lang_to, source_lang=lang_from) for x in data_df['Text']]


def lower_split(top_data_df, lang, check_lang=False):
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

    app_output.output(f"Deleted reviews due to bad content (language, no text, ..) : {lost}")
    app_output.output(f"Correct reviews : {success}")
    return result


def split_line(line, lang):
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


def remove_diacritics_from_file(filename, langs):
    with open(filename, 'r', encoding="utf-8") as f:
        for lang in langs:
            if lang == Language.CZECH.value:
                lines = [remove_diacritics_cs(line) for line in f]
            elif lang == Language.GERMAN.value:
                lines = [remove_diacritics_de(line) for line in f]

    with open(filename, 'w', encoding="utf-8") as f:
        f.writelines(lines)

    app_output.output(f"Removed diacritics from file {filename}")
