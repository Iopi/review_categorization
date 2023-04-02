import logging
import random

import fasttext
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.manifold import TSNE

import constants
import util

# now we will Create and configure logger
logging.basicConfig(filename="log.log",
                    format='%(message)s',
                    filemode='w')
# logging.basicConfig(filename="std.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')

logging.getLogger('matplotlib.font_manager').disabled = True

# Let us Create an object
logger = logging.getLogger()

# Now we are going to Set the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def output(message):
    logger.info(message)
    print(message)


def exception(message):
    logger.exception(f"\nException: {message}")
    raise Exception(message)


def plot_distribution(top_data_df, sentiment):
    """
    Plotting the sentiment distribution
    :param top_data_df:
    :return:
    """
    plt.figure()
    pd.value_counts(top_data_df[sentiment]).plot.bar(title="Sentiment General distribution in df")
    plt.xlabel(sentiment)
    plt.ylabel("No. of rows in df")
    plt.show()


def plot_category_distribution(Y_train, category_name):
    sentiment_values = pd.Series(Y_train).value_counts().sort_index()
    if len(sentiment_values) == 3:
        sns.barplot(x=np.array(['Neutral', 'Positive', 'Negative']), y=sentiment_values.values).set(title=category_name)
    else:
        sns.barplot(x=np.array(['Not annotated', 'Annotated']), y=sentiment_values.values).set(title=category_name)
    plt.show()


def get_top_data(top_data_df, top_n=5000):
    """
    Function to retrieve top few number of each category
    :param top_n:
    :return:
    """
    top_data_df_positive = top_data_df[top_data_df['General'] == 3].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['General'] == 2].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['General'] == 1].head(top_n)
    top_data_df_not_annotated = top_data_df[top_data_df['General'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral,
                                   top_data_df_not_annotated])
    return top_data_df_small


def device_recognition():
    """
    Use cuda if present
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # util.output("Device available for running: ")
    # util.output(device)
    return device


def plot_top_similar(query_word, model, limit=10, color=['maroon', 'blue']):
    embed_dim = model.wv.vectors.shape[1]
    vectors = np.empty((0, embed_dim), dtype='f')
    labels = [query_word]
    types = ['Query Word']

    vectors = np.append(vectors, model.wv.__getitem__([query_word]), axis=0)

    similar_words = model.wv.most_similar(query_word, topn=limit)
    for word, similarity in similar_words:
        vector = model.wv.__getitem__([word])
        labels.append(word)
        types.append('Similar Words')
        vectors = np.append(vectors, vector, axis=0)

    vectors_tsne = TSNE(perplexity=10, n_components=2, random_state=42, init='pca').fit_transform(vectors)
    vectors_tsne_df = pd.DataFrame({
        'X': [x for x in vectors_tsne[:, 0]],
        'Y': [y for y in vectors_tsne[:, 1]],
        'label': labels,
        'Type': types
    })

    fig = px.scatter(vectors_tsne_df, x='X', y='Y', text='label', color='Type', size_max=60,
                     color_discrete_map={'Query Word': color[0], 'Similar Words': color[1]})
    fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color, textposition='top right'))
    fig.update_layout(
        height=800,
        title_text=f't-SNE visualization for Top {limit} Similar Words to "{query_word}"'
    )

    return fig


def bin2vec(filepath):
    f = fasttext.load_model(filepath)
    filepath = filepath[:-3] + "vec"
    words = f.get_words()
    with open(filepath, 'w', encoding="utf-8") as file_out:
        file_out.write(str(len(words)) + " " + str(f.get_dimension()) + "\n")

        for w in words:
            v = f.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass


def compute_majority_class(Y_train):
    sentiment_values = pd.Series(Y_train).value_counts().sort_values(ascending=False)
    acc = sentiment_values.values[0] / sum(sentiment_values.values)
    util.output(f"MC -> accuracy: {acc}")


def compute_x(Y_train):
    sentiment_values = pd.Series(Y_train).value_counts().sort_index()
    values_total = sum(sentiment_values.values)
    val_0 = sentiment_values.values[0]
    val_1 = sentiment_values.values[1]

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for x in range(val_1):
        if random.random() < val_1 / values_total:
            TP += 1
        else:
            FP += 1

    for x in range(val_0):
        if random.random() < val_0 / values_total:
            TN += 1
        else:
            FN += 1

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print(f"MC -> recall: {recall}, precision: {precision}, accuracy: {accuracy}")


def print_similarity(vec_model_train, param):
    print(param)
    ms = vec_model_train.most_similar(param)
    print(ms)


def remove_duplicates(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        words = f.readlines()
        len_before = len(words)
        words = list(set(words))
        len_after = len(words)
        print(f"Removed duplicates {len_before - len_after}")

    with open(filename, 'w') as f:
        f.writelines(words)
