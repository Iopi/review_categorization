import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import optim

import constants
import util
from preprocessing import preprocessing_methods
from vector_model import models


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # vec_model = gensim.models.KeyedVectors.load(model_filename)
        # weights = vec_model.wv
        # # With pretrained embeddings
        # if padding:
        #     self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors),
        #                                                   padding_idx=vec_model.wv.key_to_index['pad'])
        # else:
        #     self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors))
        
        # Linear function 1: vocab_size --> 500
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 500 --> 500
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 500 --> 3
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)

        # return F.tanh(out, dim=1)
        return F.softmax(out, dim=1)


def training_FFNN(review_dict, device, X_train, Y_train_sentiment):
    vocab_size = len(review_dict)

    input_dim = vocab_size
    hidden_dim = 500  # default 500
    output_dim = 3
    # num_epochs = 1
    num_epochs = 10  # default 100

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
            bow_vec = models.make_bow_vector(review_dict, row['tokens'], device)
            # bow_vec = models.make_word2vec_vector_cnn(model, row['tokens'], device, max_sen_len)


            # Forward pass to get output
            probs = ff_nn_bow_model(bow_vec)

            # Get the target label
            target = preprocessing_methods.make_target(Y_train_sentiment[index], device)

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
            bow_vec = models.make_bow_vector(review_dict, row['tokens'], device)
            probs = ff_nn_bow_model(bow_vec)
            bow_ff_nn_predictions.append(torch.argmax(probs, dim=1).cpu().numpy()[0])
            original_lables_ff_bow.append(preprocessing_methods.make_target(Y_test_sentiment[index], device).cpu().numpy()[0])
    util.output(classification_report(original_lables_ff_bow, bow_ff_nn_predictions))
    ffnn_loss_df = pd.read_csv(ffnn_loss_file_name)
    util.output(len(ffnn_loss_df))
    util.output(ffnn_loss_df.columns)
    ffnn_plt_500_padding_100_epochs = ffnn_loss_df[' loss'].plot()
    fig = ffnn_plt_500_padding_100_epochs.get_figure()
    fig.savefig(constants.DATA_FOLDER + 'plots/' + "ffnn_bow_loss_500_padding_100_epochs_less_lr.pdf")

