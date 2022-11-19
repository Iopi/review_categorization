import pandas as pd
import matplotlib.pyplot as plt
import time
import os.path

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch.nn.functional as F

from classifires.CnnClassifier import ConvolutionalNeuralNetworkClassifier
from classifires.FFNN import FeedforwardNeuralNetModel
from classifires.LogRegClassifier import LogisticRegressionClassifier
from classifires.LSTM import LongShortTermMemory
from preprocessing import preprocessing_methods
from vector_model import models
import util
import constants

from gensim import corpora
import gensim

CREATE_MODEL = False


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


def split_train_test(top_data_df_small, class_name, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small['stemmed_tokens'],
                                                        top_data_df_small[class_name],
                                                        shuffle=shuffle_state,
                                                        test_size=test_size,
                                                        random_state=15)
    util.output("Value counts for Train sentiments")
    util.output(Y_train.value_counts())
    util.output("Value counts for Test sentiments")
    util.output(Y_test.value_counts())

    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.reset_index()
    # util.output(X_train.head())
    return X_train, X_test, Y_train, Y_test


# Function to return the dictionary either with padding word or without padding
def make_dict(top_data_df_small, padding=True):
    if padding:
        util.output("Dictionary with padded token added")
        review_dict = corpora.Dictionary([['pad']])
        review_dict.add_documents(top_data_df_small['stemmed_tokens'])
    else:
        util.output("Dictionary without padding")
        review_dict = corpora.Dictionary(top_data_df_small['stemmed_tokens'])
    return review_dict


# Function to get the util.output tensor
def make_target(label, device):
    if label == -1:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 0:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)


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


def training_CNN(model, model_filename, device, max_sen_len, X_train, Y_train_sentiment, padding=False):
    NUM_CLASSES = 3
    VOCAB_SIZE = len(model.wv)

    cnn_model = ConvolutionalNeuralNetworkClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES,
                                                     model_filename=model_filename, padding=padding)
    cnn_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)  # default 0.01
    # optimizer = optim.Adadelta(cnn_model.parameters(), lr=0.25, rho=0.9)
    # num_epochs = 1
    num_epochs = 30  # default 30

    # Open the file for writing loss
    loss_file_name = constants.DATA_FOLDER + 'plots/' + 'cnn_class_big_loss_with_padding.csv'
    f = open(loss_file_name, 'w')
    f.write('iter, loss')
    f.write('\n')
    losses = []
    cnn_model.train()
    for epoch in range(num_epochs):
        # util.output("Epoch" + str(epoch + 1))
        train_loss = 0
        for index, row in X_train.iterrows():
            # Clearing the accumulated gradients
            cnn_model.zero_grad()

            # Make the bag of words vector for stemmed tokens
            bow_vec = models.make_word2vec_vector_cnn(model, row['stemmed_tokens'], device, max_sen_len)
            # bow_vec = models.make_fasttext_vector_cnn(model, row['stemmed_tokens'], device, max_sen_len)

            # Forward pass to get util.output
            probs = cnn_model(bow_vec)

            # Get the target label
            # target = make_target(Y_train_sentiment[index], device)
            target = torch.tensor([Y_train_sentiment[index]], dtype=torch.long)

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            train_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

        # if index == 0:
        #     continue
        # util.output("Epoch ran :" + str(epoch + 1))
        f.write(str((epoch + 1)) + "," + str(train_loss / len(X_train)))
        f.write('\n')
        train_loss = 0

    torch.save(cnn_model, constants.DATA_FOLDER + 'cnn_big_model_500_with_padding.pth')

    f.close()
    # util.output("Input vector")
    # util.output(bow_vec.cpu().numpy())
    # util.output("Probs")
    # util.output(probs)
    # util.output(torch.argmax(probs, dim=1).cpu().numpy()[0])

    return cnn_model


def testing_CNN(cnn_model, word2vec_file, w2v_model, device, max_sen_len, X_test, Y_test_sentiment):
    bow_cnn_predictions = []
    original_lables_cnn_bow = []
    cnn_model.eval()
    loss_df = pd.read_csv(constants.DATA_FOLDER + 'plots/' + 'cnn_class_big_loss_with_padding.csv')
    util.output(loss_df.columns)
    # loss_df.plot('loss')
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec = models.make_word2vec_vector_cnn(w2v_model, row['stemmed_tokens'], device, max_sen_len)
            probs = cnn_model(bow_vec)
            _, predicted = torch.max(probs.data, 1)
            bow_cnn_predictions.append(predicted.cpu().numpy()[0])
            # original_lables_cnn_bow.append(make_target(Y_test_sentiment[index], device).cpu().numpy()[0])
            original_lables_cnn_bow.append(torch.tensor([Y_test_sentiment[index]], dtype=torch.long).cpu().numpy()[0])
    util.output(classification_report(original_lables_cnn_bow, bow_cnn_predictions))
    loss_file_name = constants.DATA_FOLDER + 'plots/' + 'cnn_class_big_loss_with_padding.csv'
    loss_df = pd.read_csv(loss_file_name)
    util.output(loss_df.columns)
    plt_500_padding_30_epochs = loss_df[' loss'].plot()
    fig = plt_500_padding_30_epochs.get_figure()
    fig.savefig(constants.DATA_FOLDER + 'plots/' + 'loss_plt_500_padding_30_epochs.pdf')


def training_FFNN(review_dict, device, X_train, Y_train_sentiment):
    vocab_size = len(review_dict)

    input_dim = vocab_size
    hidden_dim = 500  # default 500
    output_dim = 3
    # num_epochs = 1
    num_epochs = 100  # default 100

    ff_nn_bow_model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    ff_nn_bow_model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ff_nn_bow_model.parameters(), lr=0.01)  # default 0.01
    # optimizer = optim.Adam(ff_nn_bow_model.parameters(), lr=0.01)
    # optimizer = optim.Adadelta(ff_nn_bow_model.parameters(), lr=0.01)
    # optimizer = optim.RMSprop(ff_nn_bow_model.parameters(), lr=0.01)

    # Open the file for writing loss
    ffnn_loss_file_name = constants.DATA_FOLDER + 'ffnn_bow_class_big_loss_500_epoch_100_less_lr.csv'
    f = open(ffnn_loss_file_name, 'w')
    f.write('iter, loss')
    f.write('\n')
    losses = []
    iter = 0
    # Start training
    for epoch in range(num_epochs):
        if (epoch + 1) % 25 == 0:
            util.output("Epoch completed: " + str(epoch + 1))
        train_loss = 0
        for index, row in X_train.iterrows():
            # Clearing the accumulated gradients
            optimizer.zero_grad()

            # Make the bag of words vector for stemmed tokens
            bow_vec = models.make_bow_vector(review_dict, row['stemmed_tokens'], device)

            # Forward pass to get output
            probs = ff_nn_bow_model(bow_vec)

            # Get the target label
            target = make_target(Y_train_sentiment[index], device)

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            # Accumulating the loss over time
            train_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
        f.write(str((epoch + 1)) + "," + str(train_loss / len(X_train)))
        f.write('\n')
        train_loss = 0

    f.close()
    return ff_nn_bow_model, ffnn_loss_file_name


def testing_FFNN(review_dict, ff_nn_bow_model, ffnn_loss_file_name, device, X_test, Y_test_sentiment):
    bow_ff_nn_predictions = []
    original_lables_ff_bow = []
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec = models.make_bow_vector(review_dict, row['stemmed_tokens'], device)
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(make_target(Y_test_sentiment[index], device).cpu().numpy()[0])
    util.output(classification_report(original_lables_ff_bow, bow_ff_nn_predictions))
    ffnn_loss_df = pd.read_csv(ffnn_loss_file_name)
    util.output(len(ffnn_loss_df))
    util.output(ffnn_loss_df.columns)
    ffnn_plt_500_padding_100_epochs = ffnn_loss_df[' loss'].plot()
    fig = ffnn_plt_500_padding_100_epochs.get_figure()
    fig.savefig(constants.DATA_FOLDER + 'plots/' + "ffnn_bow_loss_500_padding_100_epochs_less_lr.pdf")


def training_LogReg(review_dict, device, X_train, Y_train_sentiment):
    VOCAB_SIZE = len(review_dict)

    #  Initialize the model
    lr_nn_model = LogisticRegressionClassifier(constants.NUM_LABELS, VOCAB_SIZE)
    lr_nn_model.to(device)

    # Loss Function
    loss_function = nn.NLLLoss()
    # loss_function = nn.CrossEntropyLoss()
    # Optimizer initlialization
    optimizer = optim.SGD(lr_nn_model.parameters(), lr=0.1)  # default 0.01
    # optimizer = optim.Adam(lr_nn_model.parameters(), lr=0.1)

    start_time = time.time()

    # Train the model
    for epoch in range(200):  # default 100
        for index, row in X_train.iterrows():
            # Step 1. Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            lr_nn_model.zero_grad()

            # Step 2. Make BOW vector for input features and target label
            bow_vec = models.make_bow_vector(review_dict, row['stemmed_tokens'], device)
            target = make_target(Y_train_sentiment[index], device)

            # Step 3. Run the forward pass.
            probs = lr_nn_model(bow_vec)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(probs, target)
            loss.backward()
            optimizer.step()
    util.output("Time taken to train the model: " + str(time.time() - start_time))
    return lr_nn_model


def testing_LogReg(review_dict, bow_nn_model, device, X_test, Y_test_sentiment):
    lr_nn_predictions = []
    original_lables = []
    start_time = time.time()
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec = models.make_bow_vector(review_dict, row['stemmed_tokens'], device)
            probs = bow_nn_model(bow_vec)
            lr_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables.append(make_target(Y_test_sentiment[index], device).cpu().numpy()[0])
    util.output(classification_report(original_lables, lr_nn_predictions))
    util.output("Time taken to predict: " + str(time.time() - start_time))


def training_Decision_Tree(Y_train_sentiment, model_filename):
    # start_time = time.time()

    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename)

    # Initialize the model
    clf_decision = DecisionTreeClassifier(random_state=2)
    # clf_decision = LinearSVC(random_state=0)

    # Fit the model
    clf_decision.fit(vectors_model, Y_train_sentiment)
    # print("Time to taken to fit the TF-IDF as input for classifier: " + str(time.time() - start_time))
    return clf_decision


def training_Linear_SVM(Y_train_sentiment, model_filename):
    # start_time = time.time()

    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename)

    # Initialize the model
    clf_svm = LinearSVC(random_state=0)

    # Fit the model
    clf_svm.fit(vectors_model, Y_train_sentiment)
    # print("Time to taken to fit the TF-IDF as input for classifier: " + str(time.time() - start_time))
    return clf_svm


def training_Logistic_Regression(Y_train_sentiment, model_filename):
    # start_time = time.time()

    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename)

    # Initialize the model
    clf = LogisticRegression(random_state=0)

    # Fit the model
    clf.fit(vectors_model, Y_train_sentiment)
    # print("Time to taken to fit the TF-IDF as input for classifier: " + str(time.time() - start_time))
    return clf


def testing_classificator_with_tfidf(clf_decision, tfidf_model, review_dict, X_test, Y_test_sentiment):
    test_features = []
    vocab_len = len(review_dict.token2id)

    # start_time = time.time()
    for index, row in X_test.iterrows():
        doc = review_dict.doc2bow(row['stemmed_tokens'])
        features = gensim.matutils.corpus2csc([tfidf_model[doc]], num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf_decision.predict(test_features)
    util.output(classification_report(Y_test_sentiment, test_predictions))
    # print("Time taken to predict using TF-IDF:" + str(time.time() - start_time))


def testing_classificator_with_bow(clf_decision, review_dict, X_test, Y_test_sentiment):
    test_features = []
    vocab_len = len(review_dict.token2id)

    # start_time = time.time()
    for index, row in X_test.iterrows():
        features = gensim.matutils.corpus2csc([review_dict.doc2bow(row['stemmed_tokens'])],
                                              num_terms=vocab_len).toarray()[:, 0]
        test_features.append(features)
    test_predictions = clf_decision.predict(test_features)
    util.output(classification_report(Y_test_sentiment, test_predictions))
    # print("Time taken to predict using TF-IDF:" + str(time.time() - start_time))


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def training_LSTM(model, word2vec_file, device, max_sen_len,
          X_train, Y_train_sentiment, padding=True, num_epochs=5,):
    # initialize running values
    train_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    eval_every = len(X_train) // 2

    lstm_model = LongShortTermMemory(word2vec_file, padding).to(device)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # training loop
    lstm_model.train()
    for epoch in range(num_epochs):
        # for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in train_loader:
        for index, row in X_train.iterrows():

            # Clearing the accumulated gradients
            lstm_model.zero_grad()

            # Make the bag of words vector for stemmed tokens
            vec = models.make_word2vec_vector_cnn(model, row['stemmed_tokens'], device, max_sen_len)
            # vec = models.make_fasttext_vector_cnn(model, row['stemmed_tokens'], device, max_sen_len)

            # Forward pass to get util.output
            probs = lstm_model(vec)

            # Get the target label
            target = torch.tensor([Y_train_sentiment[index]], dtype=torch.long)
            target = target.unsqueeze(1)
            # target = target.to(torch.float32)

            loss = loss_function(probs, target)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            train_loss += loss.item()
            # global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                lstm_model.eval()
                with torch.no_grad():
                    # validation loop
                    for (labels, (title, title_len), (text, text_len), (titletext, titletext_len)), _ in Y_train_sentiment:
                        labels = labels.to(device)
                        titletext = titletext.to(device)
                        titletext_len = titletext_len.to(device)
                        output = lstm_model(titletext, titletext_len)

                        loss = loss_function(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(Y_train_sentiment)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                lstm_model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(X_train),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(constants.DATA_FOLDER + '/model.pt', lstm_model, optimizer, best_valid_loss)
                    save_metrics(constants.DATA_FOLDER + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    # torch.save(lstm_model, constants.DATA_FOLDER + 'lstm_big_model_500_with_padding.pth')


    util.output("Input vector")
    util.output(vec.cpu().numpy())
    util.output("Probs")
    util.output(probs)
    util.output(torch.argmax(probs, dim=1).cpu().numpy()[0])
    # save_metrics(constants.DATA_FOLDER + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

    return lstm_model

def testing_LSTM(cnn_model, word2vec_file, w2v_model, device, max_sen_len, X_test, Y_test_sentiment):
    bow_cnn_predictions = []
    original_lables_cnn_bow = []
    cnn_model.eval()
    # loss_df = pd.read_csv(constants.DATA_FOLDER + 'plots/' + 'lstm_class_big_loss_with_padding.csv')
    # util.output(loss_df.columns)
    # loss_df.plot('loss')
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec = models.make_word2vec_vector_cnn(w2v_model, row['stemmed_tokens'], device, max_sen_len)
            probs = cnn_model(bow_vec)
            _, predicted = torch.max(probs.data, 1)
            bow_cnn_predictions.append(predicted.cpu().numpy()[0])
            original_lables_cnn_bow.append(torch.tensor([Y_test_sentiment[index]], dtype=torch.long).cpu().numpy()[0])
    util.output(classification_report(original_lables_cnn_bow, bow_cnn_predictions))
    # loss_file_name = constants.DATA_FOLDER + 'plots/' + 'lstm_class_big_loss_with_padding.csv'
    # loss_df = pd.read_csv(loss_file_name)
    # util.output(loss_df.columns)
    # plt_500_padding_30_epochs = loss_df[' loss'].plot()
    # fig = plt_500_padding_30_epochs.get_figure()
    # fig.savefig(constants.DATA_FOLDER + 'plots/' + 'loss_plt_500_padding_5_epochs.pdf')



def classification_sentiments(data_df_ranked, classes):
    start_time = time.time()

    # load or create word2vec model
    # word2vec_file2 = constants.DATA_FOLDER + 'models/' + constants.W2V_PAD_MODEL_NAME
    # if CREATE_MODEL:
    #     w2v_model2, word2vec_file2 = models.make_word2vec_model(data_df_ranked)
    # elif os.path.exists(word2vec_file2):
    #     w2v_model2 = gensim.models.KeyedVectors.load(word2vec_file2)
    # else:
    #     raise Exception("Word2vec model not found")

    # load or create word2vec model
    word2vec_file = constants.DATA_FOLDER + 'models/' + constants.W2V_PAD_MODEL_NAME
    if CREATE_MODEL:
        w2v_model, word2vec_file = models.make_word2vec_model(data_df_ranked)
    elif os.path.exists(word2vec_file):
        w2v_model = gensim.models.KeyedVectors.load(word2vec_file)
    else:
        raise Exception("Word2vec model not found")

    # load or create fasttext model
    # fasttext_file = constants.DATA_FOLDER + 'models/' + constants.FT_MODEL_NAME
    # if CREATE_MODEL:
    #     ft_model, fasttext_file = models.make_fasttext_model(data_df_ranked)
    # elif os.path.exists(fasttext_file):
    #     ft_model = gensim.models.KeyedVectors.load(fasttext_file)
    # else:
    #     ft_model, fasttext_file = models.make_fasttext_model(data_df_ranked)

    # print(w2v_model.wv.most_similar("objedn"))
    # print(ft_model.wv.most_similar("nasrat"))
    # print(ft_model.wv.most_similar("zabiju"))
    # print(ft_model.wv.most_similar("auto"))
    # print(ft_model.wv.most_similar("eroplan"))
    # print(ft_model.wv.most_similar("sdfsdfgfgh"))

    for class_name in classes:
        start_time_class = time.time()

        util.output("Classification sentiment " + class_name)

        # drop not needed rows
        df_sentiment = data_df_ranked[data_df_ranked[class_name] != 2]

        # Plotting the sentiment distribution
        plot_distribution(df_sentiment, class_name)

        # After selecting top few samples of each sentiment

        util.output("After segregating and taking equal number of rows for each sentiment:")
        util.output(df_sentiment[class_name].value_counts())
        # util.output(df_sentiment.head(10))

        # Call the train_test_split
        X_train, X_test, Y_train, Y_test = split_train_test(df_sentiment, class_name)

        # Use cuda if present
        device = device_recognition()


        # 1.1 # CNN with w2v model
        # max_sen_len = df_sentiment.stemmed_tokens.map(len).max()
        # cnn_model = training_CNN(w2v_model, word2vec_file, device, max_sen_len, X_train, Y_train[class_name],
        #                          padding=True)
        # testing_CNN(cnn_model, word2vec_file, w2v_model, device, max_sen_len, X_test, Y_test[class_name])

        # 1.1 # CNN with w2v model
        max_sen_len = df_sentiment.stemmed_tokens.map(len).max()
        cnn_model = training_LSTM(w2v_model, word2vec_file, device, max_sen_len, X_train, Y_train[class_name],
                                 padding=True)
        testing_LSTM(cnn_model, word2vec_file, w2v_model, device, max_sen_len, X_test, Y_test[class_name])

        # 1.2 # CNN with fasttext model
        # max_sen_len = df_sentiment.stemmed_tokens.map(len).max()
        # cnn_model = training_CNN(ft_model, fasttext_file, device, max_sen_len, X_train, Y_train[class_name])
        # testing_CNN(cnn_model, fasttext_file, ft_model, device, max_sen_len, X_test, Y_test[class_name])

        # 2 # FFNN
        # Make the dictionary without padding for the basic models
        # review_dict = make_dict(df_sentiment, padding=False)
        # ff_nn_bow_model, ffnn_loss_file_name = training_FFNN(review_dict, device, X_train, Y_train[class_name])
        # testing_FFNN(review_dict, ff_nn_bow_model, ffnn_loss_file_name, device, X_test, Y_test[class_name])

        # 3 # Logistic Regresion with BoW model
        # util.output("Logistic Regresion - Bow - pytorch")
        # review_dict = make_dict(df_sentiment, padding=False)
        # bow_nn_model = training_LogReg(review_dict, device, X_train, Y_train[class_name])
        # testing_LogReg(review_dict, bow_nn_model, device, X_test, Y_test[class_name])

        # # 5 # Decision Tree with BoW model
        # util.output("Decision Tree - Bow")
        # review_dict = make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = training_Decision_Tree(Y_train[class_name], filename)
        # testing_classificator_with_bow(clf, review_dict, X_test, Y_test[class_name])

        # # 4 # Decision Tree with Tfidf model
        # util.output("Decision Tree - Tfidf")
        # review_dict = make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_tfidf.csv'
        # tfidf_model = models.create_tfidf_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = training_Decision_Tree(Y_train[class_name], filename)
        # testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test[class_name])
        #
        # # 6 # Linear SVM with BoW model
        # util.output("Linear SVM - BoW")
        # review_dict = make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = training_Linear_SVM(Y_train[class_name], filename)
        # testing_classificator_with_bow(clf, review_dict, X_test, Y_test[class_name])
        #
        # # 7 # Linear SVM with Tfidf model
        # util.output("Linear SVM - Tfidf")
        # review_dict = make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_tfidf.csv'
        # tfidf_model = models.create_tfidf_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = training_Linear_SVM(Y_train[class_name], filename)
        # testing_classificator_with_tfidf(clf, tfidf_model, review_dict, X_test, Y_test[class_name])
        #
        # # 8 # Logistic Regression with BoW model
        # util.output("Logistic Regression - Bow")
        # review_dict = make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = training_Logistic_Regression(Y_train[class_name], filename)
        # testing_classificator_with_bow(clf, review_dict, X_test, Y_test[class_name])
        #
        # # 9 # Logistic Regression with Tfidf model
        # util.output("Logistic Regression - Tfidf")
        # review_dict = make_dict(df_sentiment, padding=False)
        # filename = constants.DATA_FOLDER + 'train_review_bow.csv'
        # models.create_bow_model_file(review_dict, df_sentiment, X_train, filename)
        # clf = training_Logistic_Regression(Y_train[class_name], filename)
        # testing_classificator_with_tfidf(clf, review_dict, X_test, Y_test[class_name])

        util.output("Time taken to predict " + class_name + " :" + str(time.time() - start_time_class))
        # break
    util.output("Time taken to predict all:" + str(time.time() - start_time))


def create_model():
    top_data_df = pd.read_excel(constants.DATA_FOLDER + 'feed_cs_model.xlsx', sheet_name="Sheet1")
    # Tokenize the text column to get the new column 'tokenized_text'
    preprocessing_methods.tokenization(top_data_df)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(top_data_df)

    # creating model
    # models.make_word2vec_model(top_data_df, padding=False)
    # models.make_word2vec_model(top_data_df, padding=True)
    models.make_fasttext_model(top_data_df)


def main():
    # only creating model
    create_model()

    # Example of target with class indices
    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # output = loss(input, target)
    # output.backward()
    #
    #  # Example of target with class probabilities
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randn(3, 5).softmax(dim=1)
    # output = loss(input, target)
    # output.backward()
    #
    # m = nn.Sigmoid()
    # loss = nn.BCELoss()
    # input = torch.randn(3, requires_grad=True)
    # target = torch.empty(3).random_(2)
    # output = loss(m(input), target)
    # output.backward()

    top_data_df = pd.read_excel(constants.DATA_FOLDER + constants.REVIEWS_DATA_NAME, sheet_name="Sheet1", nrows=550)
    top_data_df = top_data_df.dropna(thresh=4)
    # util.output("Columns in the original dataset:\n")
    # util.output(top_data_df.columns)

    # Removing the stop words
    # preprocessing.remove_stopwords()

    # Tokenize the text column to get the new column 'tokenized_text'
    preprocessing_methods.tokenization(top_data_df)

    # Get the stemmed_tokens
    preprocessing_methods.stemming(top_data_df)

    classes = top_data_df.columns[1:10]

    # temp_data = top_data_df.copy()
    # # annotated 1, not annotated 0
    # map_sentiment_annotated(temp_data)
    # classification_sentiments(temp_data, classes)

    temp_data = top_data_df.copy()
    # positive 1, negative and neutral 0
    map_sentiment_positive(temp_data)
    classification_sentiments(temp_data, classes)
    #
    # temp_data = top_data_df.copy()
    # # negative 1, positive and neutral 0
    # map_sentiment_negative(temp_data)
    # classification_sentiments(temp_data, classes)

    # map_sentiment_annotate(top_data_df)
    # map_sentiment(top_data_df)


if __name__ == "__main__":
    main()
