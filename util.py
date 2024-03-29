import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from view import app_output


def plot_distribution(data_df, category_name):
    """
    Plotting the distribution of category
    :param data_df: dataframe
    :param category_name: category name
    :return:
    """
    plt.figure()
    pd.value_counts(data_df[category_name]).plot.bar(title=f"{category_name} distribution in data")
    plt.xlabel(category_name)
    plt.ylabel("No. of rows in data")
    plt.show()


def sentiment_count(data_df, category_name):
    """
    Count positive and negative sentiment in data for category
    :param data_df: data dataframe
    :param category_name: category name
    :return:
    """
    category_data = data_df[category_name]
    values = category_data.values
    positive_count = np.count_nonzero(values == 1)
    negative_count = np.count_nonzero(values == 0)

    app_output.output(
        f"{category_name} - Positive {positive_count} / Negative {negative_count} (sum {positive_count + negative_count})")


def plot_category_distribution(Y_train, category_name):
    """
    Plot category distribution
    :param Y_train: train labels
    :param category_name: category name
    :return:
    """
    sentiment_values = pd.Series(Y_train).value_counts().sort_index()
    if len(sentiment_values) == 3:
        sns.barplot(x=np.array(['Neutral', 'Positive', 'Negative']), y=sentiment_values.values).set(title=category_name)
    else:
        sns.barplot(x=np.array(['Not annotated', 'Annotated']), y=sentiment_values.values).set(title=category_name)
    plt.show()


def device_recognition():
    """
    Detect cuda or cpu device
    :return: device (cpu/gpu)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def plot_top_similar(query_word, model, limit=10, color=['maroon', 'blue']):
    """
    Plot some most similar words to query word in vector model
    :param query_word: query word
    :param model: vector model
    :param limit: number of words
    :param color: colors of plot
    :return: figure
    """
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


def compute_majority_class(Y_train):
    """
    Compute of majority class
    :param Y_train: train labels
    :return:
    """
    sentiment_values = pd.Series(Y_train).value_counts().sort_values(ascending=False)
    acc = sentiment_values.values[0] / sum(sentiment_values.values)
    app_output.output(f"Majority class: {acc}")


def print_metrics(true_labels, classified_labels, category_name):
    """
    Print metrics
    :param true_labels: true labels
    :param classified_labels: classified labels
    :param category_name: category name
    :return:
    """
    if len(true_labels) != len(classified_labels):
        app_output.exception("Count of true labels must be same as classified labels.")

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for x in range(len(true_labels)):
        if true_labels[x] == classified_labels[x]:
            if classified_labels[x] == 1:
                TP += 1
            else:
                TN += 1

        else:
            if classified_labels[x] == 1:
                FP += 1
            else:
                FN += 1
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0
    try:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except ZeroDivisionError:
        accuracy = 0
    try:
        f1score = 2 * precision * recall / (precision + recall)
        f1score2 = (2 * TP) / (2 * TP + FP + FN)
    except ZeroDivisionError:
        f1score = 0
        f1score2 = 0

    app_output.output(
        f"{category_name} -> recall: {recall}, precision: {precision}, accuracy: {accuracy}, f1-score {f1score} / {f1score2}")


def print_similarity(vec_model, param):
    """
    Print similarity of word in model
    :param vec_model: vector model
    :param param: word
    :return:
    """
    app_output.output(param)
    ms = vec_model.most_similar(param)
    app_output.output(ms)


def remove_duplicates(filename):
    """
    Remove duplicate words in file
    :param filename: filename
    :return:
    """
    with open(filename, 'r', encoding="utf-8") as f:
        words = f.readlines()
        len_before = len(words)
        words = list(set(words))
        len_after = len(words)
        app_output.output(f"Removed duplicates {len_before - len_after}")

    with open(filename, 'w') as f:
        f.writelines(words)


def print_info(args, is_fasttext):
    """
    Print info about classification - vector model and classification model
    :param args: user arguments
    :param is_fasttext: if vector model is fasttext
    :return:
    """
    info = ""
    if is_fasttext is None:
        if args.model_type == 'bow':
            info += "Bag of words - "
        elif args.model_type == 'tfidf':
            info += "Tf-idf - "
        else:
            app_output.exception(f"Wrong model type {args.model_type}")

        if args.classi_model == "svm":
            info += 'Support vector machines'

        elif args.classi_model == "logreg":
            info += 'Logistic regression'

        elif args.classi_model == "dectree":
            info += 'Decision tree'

        else:
            app_output.exception(f"Wrong classification model {args.classi_model}")

    else:
        if is_fasttext:
            info += "Fasttext - "
        else:
            info += "Word2vec - "

        if args.classi_model == "lstm":
            app_output.output(info + "Long short-term memory")

        elif args.classi_model == "cnn":
            app_output.output(info + "Convolutional neural networks")

        else:
            app_output.output(f"Wrong classification model {args.classi_model}")


def compare_reviews(filename_1, filename_2, n_rows):
    """
    Compare two files of reciews and compute metrics
    :param filename_1: filename 1
    :param filename_2: filename 2
    :param n_rows: Compare top n rows
    :return:
    """
    reviews_1 = pd.read_excel(filename_1, sheet_name="Sheet1", nrows=n_rows)
    reviews_2 = pd.read_excel(filename_2, sheet_name="Sheet1", nrows=n_rows)
    categories = reviews_1.columns[1:10]

    for category_name in categories:
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for x in range(len(reviews_1[category_name])):
            label_1 = reviews_1[category_name][x]
            label_2 = reviews_2[category_name][x]

            if label_1 is not np.NaN:
                if label_2 is not np.NaN:
                    TP += 1
                else:
                    FP += 1
            else:
                if label_2 is not np.NaN:
                    FN += 1
                else:
                    TN += 1

        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        try:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        except ZeroDivisionError:
            accuracy = 0
        try:
            f1score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1score = 0

        app_output.output(
            f"{category_name} category compare -> recall: {recall}, precision: {precision}, accuracy: {accuracy}, f1-score {f1score}")
        app_output.output(f"+ {TP + TN}, - {FP + FN}")

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for x in range(len(reviews_1[category_name])):
            label_1 = reviews_1[category_name][x]
            label_2 = reviews_2[category_name][x]

            if label_1 == "Positive":
                if label_2 == "Positive":
                    TP += 1
                elif label_2 == "Negative":
                    FP += 1
            elif label_1 == "Negative":
                if label_2 == "Positive":
                    FN += 1
                elif label_2 == "Negative":
                    TN += 1

        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        try:
            accuracy = (TP + TN) / (TP + TN + FP + FN)
        except ZeroDivisionError:
            accuracy = 0
        try:
            f1score = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1score = 0

        app_output.output(
            f"{category_name} sentiment compare -> recall: {recall}, precision: {precision}, accuracy: {accuracy}, f1-score {f1score}")
        app_output.output(f"+ {TP + TN}, - {FP + FN}\n")
