import constants
import gensim
import pandas as pd
import torch
import util
from gensim import corpora
from gensim.models import FastText
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from preprocessing import preprocessing_methods
from sklearn.metrics import classification_report


def create_lower_split_model(args):
    top_data_df = pd.read_excel(args.feed_path, sheet_name="Sheet1")

    # top_data_df_2 = pd.read_excel("data/feed/feed_en_2.xlsx", sheet_name="Sheet1")
    # top_data_df = pd.concat([top_data_df, top_data_df_2], axis=0)
    # top_data_df = top_data_df.reset_index(drop=True)

    result = preprocessing_methods.lower_split(top_data_df, args.lang, check_lang=False)
    preprocessing_methods.remove_bad_words(result, args.lang)
    len_before = len(result)
    result = [x for x in result if x != ['']]
    len_after = len(result)
    print(f"Before {len_before} and after {len_after}, diff -> {len_before - len_after}")
    if args.model_type == 'ft':
        make_fasttext_model(result, fasttext_file=args.model_path)
    elif args.model_type == 'w2v':
        make_word2vec_model(result, word2vec_file=args.model_path)
    else:
        util.exception(f"Model type {args.model_type} not found.")


def make_fasttext_model(temp_df, sg=1, min_count=2, vector_size=300, workers=3, window=5, fasttext_file=None):
    ft_model = FastText(sg=sg, vector_size=vector_size, window=window, min_count=min_count, workers=workers,
                        sentences=temp_df)

    if fasttext_file is None:
        fasttext_file = constants.DATA_FOLDER + 'vec_model/' + 'fasttext_' + str(vector_size) + "_" + '.bin'
    ft_model.save(fasttext_file)

    return ft_model, fasttext_file


def make_word2vec_model(temp_df, padding=True, sg=1, min_count=2, vector_size=300, workers=3, window=3,
                        word2vec_file=None):
    if padding:
        temp_df = pd.Series(temp_df).values
        temp_df = list(temp_df)
        temp_df.append(['pad'])
        if word2vec_file is None:
            word2vec_file = constants.DATA_FOLDER + 'vec_model/' + 'word2vec_' + str(vector_size) + '_PAD.bin'
    else:
        if word2vec_file is None:
            word2vec_file = constants.DATA_FOLDER + 'vec_model/' + 'word2vec_' + str(vector_size) + '.bin'
    w2v_model = Word2Vec(temp_df, min_count=min_count, vector_size=vector_size, workers=workers, window=window, sg=sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file


def make_vector_index_map(model, sentence, max_sen_len, is_fasttext):
    sentence_len = len(sentence)
    i = max_sen_len - sentence_len

    if not is_fasttext:
        padding_idx = model.key_to_index['pad']
        sentence_vec = [padding_idx for i in range(max_sen_len)]
    else:
        sentence_vec = [0] * max_sen_len

    for word in sentence:
        if word not in model.key_to_index:
            # in case of fasttext, we can use most similar vector
            if is_fasttext:
                sentence_vec[i] = model.key_to_index[model.most_similar(word)[0][0]]
            else:
                sentence_vec[i] = 0
        else:
            sentence_vec[i] = model.key_to_index[word]
        i += 1

    return sentence_vec


def make_vector_index_map_cnn(model, sentence, device, max_sen_len, is_fasttext):
    if not is_fasttext:
        padding_idx = model.key_to_index['pad']
        sentence_vec = [padding_idx for i in range(max_sen_len)]
    else:
        sentence_vec = [0] * max_sen_len

    i = 0
    for word in sentence:
        if word not in model.key_to_index:
            if is_fasttext:
                sentence_vec[i] = model.key_to_index[model.most_similar(word)[0][0]]
            else:
                sentence_vec[i] = 0
        else:
            sentence_vec[i] = model.key_to_index[word]
        i += 1
    return torch.tensor(sentence_vec, dtype=torch.long, device=device).view(1, -1)


def create_tfidf_model_file(review_dict, df_sentiment, X_train, filename):
    # BOW corpus is required for tfidf model
    corpus = [review_dict.doc2bow(line) for line in df_sentiment['tokens']]
    # TF-IDF Model
    tfidf_model = TfidfModel(corpus)
    # Storing the tfidf vectors for training data in a file
    vocab_len = len(review_dict.token2id)
    with open(filename, 'w+', encoding='utf-8') as tfidf_file:
        for index, row in X_train.items():
            doc = review_dict.doc2bow(row)
            features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                header = ";".join(str(review_dict[ele]) for ele in range(vocab_len))
                header = header.replace("\n", "")
                tfidf_file.write(header)
                tfidf_file.write("\n")
            line1 = ";".join([str(vector_element) for vector_element in features])
            tfidf_file.write(line1)
            tfidf_file.write('\n')

    return tfidf_model


def create_bow_model_file(review_dict, df_sentiment, X_train, filename):
    vocab_len = len(review_dict)
    with open(filename, 'w+', encoding='utf-8') as bow_file:
        for index, row in X_train.items():
            features = gensim.matutils.corpus2csc([review_dict.doc2bow(row)],
                                                  num_terms=vocab_len).toarray()[:, 0]
            if index == 0:
                header = ";".join(str(review_dict[ele]) for ele in range(vocab_len))
                header = header.replace("\n", "")
                bow_file.write(header)
                bow_file.write("\n")
            line1 = ";".join([str(vector_element) for vector_element in features])
            bow_file.write(line1)
            bow_file.write('\n')


# Function to return the dictionary either with padding word or without padding
def make_dict(top_data_df_small, padding=True):
    if padding:
        review_dict = corpora.Dictionary([['pad']])
        review_dict.add_documents(top_data_df_small['tokens'])
    else:
        review_dict = corpora.Dictionary(top_data_df_small['tokens'])
    return review_dict


def testing_classificator_with_tfidf(clf_decision, tfidf_model, review_dict, X_test, Y_test_sentiment):
    test_features = []
    vocab_len = len(review_dict.token2id)

    for index, row in X_test.items():
        doc = review_dict.doc2bow(row)
        features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf_decision.predict(test_features)
    util.output(classification_report(Y_test_sentiment, test_predictions))


def testing_classificator_with_bow(clf_decision, review_dict, X_test, Y_test_sentiment):
    test_features = []
    vocab_len = len(review_dict.token2id)

    for index, row in X_test.items():
        features = gensim.matutils.corpus2csc([review_dict.doc2bow(row)],
                                              num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf_decision.predict(test_features)
    util.output(classification_report(Y_test_sentiment, test_predictions))
