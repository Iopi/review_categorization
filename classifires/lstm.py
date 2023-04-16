import numpy as np
import torch.nn as nn
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim
import torch
from torch.utils.data import TensorDataset, DataLoader

import constants
import util
from models import model_methods


class LongShortTermMemory(nn.Module):
    def __init__(self, device, no_layers, hidden_dim, output_dim, embedding_dim, drop_prob=0.5,
                 model_filename_train=None, model_filename_test=None, trans_matrix=None):
        super(LongShortTermMemory, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        # self.vocab_size = vocab_size

        # embedding and LSTM layers
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)

        vec_model_train = gensim.models.KeyedVectors.load(model_filename_train)
        weights_train = vec_model_train.wv

        self.embedding_train = nn.Embedding.from_pretrained(torch.FloatTensor(weights_train.vectors))

        if model_filename_test:
            vec_model_test = gensim.models.KeyedVectors.load(model_filename_test)
            weights_test = vec_model_test.wv
            self.embedding_test = nn.Embedding.from_pretrained(torch.FloatTensor(weights_test.vectors))
        else:
            self.embedding_test = self.embedding_train
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors),
        #                                                   padding_idx=vec_model.wv.key_to_index['pad'])

        # lstm
        # self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
        #                     num_layers=no_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size=weights_train.vector_size, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        # (input_size=embedding_dim, hidden_size=self.hidden_dim,
        #                             num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        self.sig2 = nn.Softmax()

        self.trans_matrix = None if trans_matrix is None else torch.from_numpy(trans_matrix).float().to(device)

    def forward(self, x, hidden, train_input):
        batch_size = x.size(0)
        # embeddings and lstm_out
        if train_input:
            embeds = self.embedding_train(x)  # shape: B x S x Feature   since batch = True
        else:
            embeds = self.embedding_test(x)  # shape: B x S x Feature   since batch = True
            if self.trans_matrix is not None:
                for i in range(len(embeds)):
                    embeds[i] = torch.matmul(self.trans_matrix, embeds[i].T).T

        # print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)

        # sig_out2 = self.sig2(out)
        # sig_out2 = sig_out2.view(batch_size, -1)
        # sig_out2 = sig_out2[:, -1]

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        #
        sig_out = sig_out[:, -1]  # get last batch of labels
        # sig_out = out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


def training_LSTM(vec_model, trans_matrix, device, max_sen_len, X_train, Y_train_sentiment, binary, is_fasttext,
                  batch_size=1, model_filename_train=None, model_filename_test=None):
    X_train = [model_methods.make_vector_index_map(vec_model, line, max_sen_len, is_fasttext) for line in X_train]
    X_train = np.array(X_train)
    Y_train = Y_train_sentiment.to_numpy()
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, shuffle=True, test_size=0.25,
                                                          random_state=15)
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(Y_valid))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    no_layers = 2
    # vocab_size = len(vocab) + 1  # extra 1 for padding
    embedding_dim = 64
    output_dim = 3
    hidden_dim = 256

    lstm_model = LongShortTermMemory(device, no_layers, hidden_dim, output_dim, embedding_dim, drop_prob=0.5,
                                     model_filename_train=model_filename_train,
                                     model_filename_test=model_filename_test,
                                     trans_matrix=trans_matrix)

    # moving to gpu
    lstm_model.to(device)

    # loss and optimization functions
    lr = 0.001     # default 0.001
    # if binary:
    #     criterion = nn.BCELoss()
    # else:
    #     criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

    # function to predict accuracy
    def acc(pred, label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    clip = 5
    epochs = 5
    valid_loss_min = np.Inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_acc = 0.0
        lstm_model.train()
        # initialize hidden state
        h = lstm_model.init_hidden(batch_size, device)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            lstm_model.zero_grad()
            output, h = lstm_model(inputs, h, True)

            # calculate the loss and perform backprop
            # loss = criterion(output.squeeze(), labels.float())
            loss = criterion(output, labels.float())  # if batch_size = 1
            loss.backward()
            train_losses.append(loss.item())
            # calculating accuracy
            accuracy = acc(output, labels)
            train_acc += accuracy
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(lstm_model.parameters(), clip)
            optimizer.step()

        val_h = lstm_model.init_hidden(batch_size, device)
        val_losses = []
        val_acc = 0.0
        lstm_model.eval()
        for inputs, labels in valid_loader:
            val_h = tuple([each.data for each in val_h])

            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = lstm_model(inputs, val_h, True)
            # val_loss = criterion(output.squeeze(), labels.float())
            val_loss = criterion(output, labels.float())  # if batch_size = 1

            val_losses.append(val_loss.item())

            accuracy = acc(output, labels)
            val_acc += accuracy

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_val_acc = val_acc / len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        # print(f'Epoch {epoch + 1}')
        # print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        # print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
        if epoch_val_loss <= valid_loss_min:
            torch.save(lstm_model.state_dict(), constants.DATA_FOLDER + 'state_dict.pt')
            # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
            #                                                                                 epoch_val_loss))
            valid_loss_min = epoch_val_loss
        # print(25 * '==')

    fig = plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train loss')
    plt.plot(epoch_vl_loss, label='Validation loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()

    # plt.show()

    return lstm_model


def testing_LSTM(lstm_model, vec_model_test, device, max_sen_len, X_test, Y_test_sentiment, is_fasttext, category_name):
    bow_cnn_predictions = []
    lstm_model.eval()
    with torch.no_grad():
        i = 0
        for tags in X_test:
            # print(text)
            # tokens = simple_preprocess(text, deacc=True)
            # tags = [czech_stemmer.cz_stem(word) for word in tokens]
            vec = model_methods.make_vector_index_map(vec_model_test, tags, max_sen_len, is_fasttext)

            inputs = np.expand_dims(vec, axis=0)
            torch.from_numpy(inputs).float().to(device)
            inputs = torch.from_numpy(inputs).to(device)
            batch_size=1
            h = lstm_model.init_hidden(batch_size, device)
            h = tuple([each.data for each in h])
            output, h = lstm_model(inputs, h, False)
            if output < 0.5:
                bow_cnn_predictions.append(0)
            else:
                bow_cnn_predictions.append(1)
            i += 1
    # util.print_metrics(Y_test_sentiment, bow_cnn_predictions, category_name)
    util.output(classification_report(Y_test_sentiment, bow_cnn_predictions))


def classifie_LSTM(lstm_model, source_model, device, max_sen_len, words, is_fasttext=True):
    lstm_model.eval()
    with torch.no_grad():
        vec = model_methods.make_vector_index_map(source_model, words, max_sen_len, is_fasttext)

        inputs = np.expand_dims(vec, axis=0)
        torch.from_numpy(inputs).float().to(device)
        inputs = torch.from_numpy(inputs).to(device)
        batch_size = 1
        h = lstm_model.init_hidden(batch_size, device)
        h = tuple([each.data for each in h])
        output, h = lstm_model(inputs, h, False)

        return output
