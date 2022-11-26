import logging

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

# now we will Create and configure logger
logging.basicConfig(filename="std.log",
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
    sentiment_values = pd.Series(Y_train[category_name]).value_counts().sort_index()
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
