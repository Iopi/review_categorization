import torch
from sklearn.model_selection import train_test_split

import util
from preprocessing import czech_stemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer


def remove_bad_words():
    """
    Removing the stop words
    :return:
    """
    util.output(remove_stopwords("Restaurant had a really good service!!"))
    util.output(remove_stopwords("I did not like the food!!"))
    util.output(remove_stopwords("This product is not good!!"))


def tokenization(top_data_df_small):
    """
    Tokenize the text column to get the new column 'tokenized_text'
    :param top_data_df_small:
    :return:
    """
    top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['Text']]
    # util.output(top_data_df_small['tokenized_text'].head(10))


def stemming(top_data_df_small, lang):
    if lang == 'cs':

        # tagger = Tagger("D:\skola\Diplomka\data\morpho\czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger")
        # tokens = list(tagger.tag(top_data_df_small['stemmed_tokens'], convert ='strip_lemma_id'))
        # top_data_df_small['stemmed_tokens'] = tokens
        top_data_df_small['stemmed_tokens'] = [[czech_stemmer.cz_stem(word) for word in tokens] for tokens in
                                               top_data_df_small['tokenized_text']]
        # util.output(top_data_df_small['stemmed_tokens'].head(10))

    else:
        porter = PorterStemmer()
        top_data_df_small['stemmed_tokens'] = [[porter.stem(word) for word in tokens] for tokens in
                                               top_data_df_small['tokenized_text']]



def get_rank_id_annotated(sentiment):
    if sentiment == "Positive" or sentiment == "Negative" or sentiment == "Neutral":
        return 1
    else:
        return 0


def get_rank_id_positive(sentiment):
    if sentiment == "Positive":
        return 1
    elif sentiment == "Negative" or sentiment == "Neutral":
        return 0
    else:
        return 2


def get_rank_id_negative(sentiment):
    if sentiment == "Negative":
        return 1
    elif sentiment == "Positive" or sentiment == "Neutral":
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
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small['stemmed_tokens'],
                                                        top_data_df_small[class_name],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    # util.output("Value counts for Train sentiments")
    # util.output(Y_train.value_counts())
    # util.output("Value counts for Test sentiments")
    # util.output(Y_test.value_counts())

    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.reset_index()
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

