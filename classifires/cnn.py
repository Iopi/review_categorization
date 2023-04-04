

import gensim
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, recall_score, accuracy_score, precision_score, f1_score
from torch import optim

import constants
import util
from models import model_methods

# import torch.optim as optim

EMBEDDING_SIZE = 500
NUM_FILTERS = 10


class ConvolutionalNeuralNetworkClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, model_filename, device, window_sizes=(1,2,3,5), padding=False,
                 trans_matrix=None, model_filename_test=None):
        super(ConvolutionalNeuralNetworkClassifier, self).__init__()
        vec_model = gensim.models.KeyedVectors.load(model_filename)
        weights = vec_model.wv
        # With pretrained embeddings
        if padding:
            self.embedding_train = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=vec_model.wv.key_to_index['pad'])
        else:
            self.embedding_train = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors))

        if model_filename_test:
            vec_model_test = gensim.models.KeyedVectors.load(model_filename_test)
            weights_test = vec_model_test.wv

            self.embedding_test = nn.Embedding.from_pretrained(torch.FloatTensor(weights_test.vectors))
        else:
            self.embedding_test = self.embedding_train
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [window_size, vec_model.vector_size], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

        # embedding_dim = 300
        # hidden_dim = 200
        # output_dim=1
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(in_channels=embedding_dim, out_channels=4, kernel_size=fs)
        #     for fs in window_sizes
        # ])
        # self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        # self.fc = nn.Linear(len(window_sizes) * 4, hidden_dim)
        # self.out = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(0.5)

        self.trans_matrix = None if trans_matrix is None else torch.from_numpy(trans_matrix).float().to(device)

    # def forward(self, x, train_input):
    #     if train_input:
    #         embedded = self.embedding_train(x)  # shape: B x S x Feature   since batch = True
    #     else:
    #         embedded = self.embedding_test(x)  # shape: B x S x Feature   since batch = True
    #         if self.trans_matrix is not None:
    #             for i in range(len(embedded)):
    #                 embedded[i] = torch.matmul(self.trans_matrix, embedded[i].T).T
    #     embedded = embedded.permute(0, 2, 1)
    #     conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
    #     pooled = [self.pool(conv).squeeze(-1) for conv in conved]
    #     cat = self.dropout(torch.cat(pooled, dim=1))
    #     hidden = self.dropout(nn.functional.relu(self.fc(cat)))
    #     output = self.out(hidden)
    #     return output


    def forward(self, x, train_input):
        # x = self.embedding(x) # [B, T, E]

        if train_input:
            x = self.embedding_train(x)  # shape: B x S x Feature   since batch = True
        else:
            x = self.embedding_test(x)  # shape: B x S x Feature   since batch = True
            if self.trans_matrix is not None:
                for i in range(len(x)):
                    x[i] = torch.matmul(self.trans_matrix, x[i].T).T

        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        probs = F.softmax(logits, dim = 1)
        # probs = F.sigmoid(logits)
        # probs = probs.view(1, -1)
        # probs = probs[:, -1]  # get last batch of labels

        # return logits
        return probs


def training_CNN(model, model_filename, trans_matrix, device, max_sen_len, X_train, Y_train_sentiment, binary,
                 padding=False, model_filename_test=None):
    if binary:
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 3
    VOCAB_SIZE = len(model)
    # VOCAB_SIZE = len(model.wv)

    cnn_model = ConvolutionalNeuralNetworkClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES,
                                                     model_filename=model_filename, device=device, padding=padding,
                                                     trans_matrix=trans_matrix, model_filename_test=model_filename_test)
    cnn_model.to(device)
    # if binary:
    #     loss_function = nn.BCELoss()
    # else:
    #     loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCELoss()
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
        for index, row in X_train.items():
            # Clearing the accumulated gradients
            cnn_model.zero_grad()

            # Make the bag of words vector for stemmed tokens
            bow_vec = model_methods.make_word2vec_vector_cnn(model, row, device, max_sen_len)
            # bow_vec = models.make_fasttext_vector_cnn(model, row['tokens'], device, max_sen_len)

            # Forward pass to get util.output
            probs = cnn_model(bow_vec, True)

            # Get the tarLongShortTermMemoryget label
            # target = make_target(Y_train_sentiment[index], device)
            target = torch.tensor([Y_train_sentiment[index]], dtype=torch.long, device=device)

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


def testing_CNN(cnn_model, w2v_model, device, max_sen_len, X_test, Y_test_sentiment):
    bow_cnn_predictions = []
    original_lables_cnn_bow = []
    cnn_model.eval()
    loss_df = pd.read_csv(constants.DATA_FOLDER + 'plots/' + 'cnn_class_big_loss_with_padding.csv')
    util.output(loss_df.columns)
    # loss_df.plot('loss')
    with torch.no_grad():
        for index, row in X_test.items():
            bow_vec = model_methods.make_word2vec_vector_cnn(w2v_model, row, device, max_sen_len)
            probs = cnn_model(bow_vec, False)
            _, predicted = torch.max(probs.data, 1)
            bow_cnn_predictions.append(predicted.cpu().numpy()[0])
            # original_lables_cnn_bow.append(make_target(Y_test_sentiment[index], device).cpu().numpy()[0])
            original_lables_cnn_bow.append(torch.tensor([Y_test_sentiment[index]], dtype=torch.long).cpu().numpy()[0])
    util.output(classification_report(original_lables_cnn_bow, bow_cnn_predictions))
    # util.output("recall " + str(recall_score(original_lables_cnn_bow, bow_cnn_predictions)))
    # util.output("accuracy " + str(accuracy_score(original_lables_cnn_bow, bow_cnn_predictions)))
    # util.output("precision " + str(precision_score(original_lables_cnn_bow, bow_cnn_predictions)))
    # util.output("f1_score " + str(f1_score(original_lables_cnn_bow, bow_cnn_predictions)))
    loss_file_name = constants.DATA_FOLDER + 'plots/' + 'cnn_class_big_loss_with_padding.csv'
    loss_df = pd.read_csv(loss_file_name)
    util.output(loss_df.columns)
    plt_500_padding_30_epochs = loss_df[' loss'].plot()
    fig = plt_500_padding_30_epochs.get_figure()
    fig.savefig(constants.DATA_FOLDER + 'plots/' + 'loss_plt_500_padding_30_epochs.pdf')

