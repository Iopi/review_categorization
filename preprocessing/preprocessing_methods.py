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


def stemming(top_data_df_small):
    # tagger = Tagger("D:\skola\Diplomka\data\morpho\czech-morfflex-pdt-161115/czech-morfflex-pdt-161115.tagger")
    # tokens = list(tagger.tag(top_data_df_small['stemmed_tokens'], convert ='strip_lemma_id'))
    # top_data_df_small['stemmed_tokens'] = tokens
    top_data_df_small['stemmed_tokens'] = [[czech_stemmer.cz_stem(word) for word in tokens] for tokens in
                                           top_data_df_small['tokenized_text']]
    # util.output(top_data_df_small['stemmed_tokens'].head(10))





