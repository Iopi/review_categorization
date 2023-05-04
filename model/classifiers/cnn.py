import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import optim

from view import app_output
from controller import vector_reprezentation


class ConvolutionalNeuralNetworkClassifier(nn.Module):
    """
    Convolutional Neural Network Classifier
    """
    def __init__(self, num_classes, model_filename, device, window_sizes=(1, 2, 3, 5), padding=False,
                 trans_matrix=None, model_filename_test=None):
        """
        Initialization of CNN
        :param num_classes: number of classes
        :param model_filename: vector model filename
        :param device: device (cpu/gpu)
        :param window_sizes: window sizes for filter application
        :param padding:if padding is used
        :param trans_matrix: transformation matrix if exists
        :param model_filename_test: vector model filename for test if exists
        """
        super(ConvolutionalNeuralNetworkClassifier, self).__init__()

        num_filters = 10
        vec_model = gensim.models.KeyedVectors.load(model_filename)
        weights = vec_model.wv

        if padding:
            self.embedding_train = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors),
                                                                padding_idx=vec_model.wv.key_to_index['pad'])
        else:
            self.embedding_train = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors))

        if model_filename_test:
            vec_model_test = gensim.models.KeyedVectors.load(model_filename_test)
            weights_test = vec_model_test.wv

            self.embedding_test = nn.Embedding.from_pretrained(torch.FloatTensor(weights_test.vectors))
        else:
            self.embedding_test = self.embedding_train

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, vec_model.vector_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)

        self.trans_matrix = None if trans_matrix is None else torch.from_numpy(trans_matrix).float().to(device)

    def forward(self, x, train_input):
        """
        Passing input data to CNN
        :param x: input data
        :param train_input: true if training, False if testing
        :return: probability of positive sentiment
        """
        if train_input:
            x = self.embedding_train(x)
        else:
            x = self.embedding_test(x)
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

        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        probs = F.softmax(logits, dim=1)

        return probs


def training_CNN(model, model_filename, trans_matrix, device, max_sen_len, X_train, Y_train, is_fasttext,
                 padding=False, model_filename_test=None):
    """
    Training of CNN model
    :param model: vector model
    :param model_filename: vector model filename
    :param trans_matrix: transformation matrix
    :param device: device (cpu/gpu)
    :param max_sen_len: maximal text length in data
    :param X_train: train tokens
    :param Y_train: train labels
    :param is_fasttext: if vector model is fasttext
    :param padding: if padding is used
    :param model_filename_test: vector model filename for test
    :return: trained CNN model
    """
    num_classes = 2

    cnn_model = ConvolutionalNeuralNetworkClassifier(num_classes=num_classes, model_filename=model_filename,
                                                     device=device, padding=padding, trans_matrix=trans_matrix,
                                                     model_filename_test=model_filename_test)
    cnn_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    num_epochs = 30

    cnn_model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for index, row in X_train.items():
            cnn_model.zero_grad()
            bow_vec = vector_reprezentation.make_vector_index_map_cnn(model, row, device, max_sen_len, is_fasttext)
            # Forward pass to get probability of positive sentiment
            probs = cnn_model(bow_vec, True)
            # get label
            target = torch.tensor([Y_train[index]], dtype=torch.long, device=device)
            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            train_loss += loss.item()
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

    return cnn_model


def testing_CNN(cnn_model, vec_model, device, max_sen_len, X_test, Y_test, is_fasttext):
    """
    Testing of CNN model
    :param cnn_model: cnn model
    :param vec_model: vector model
    :param device: device (cpu/gpu)
    :param max_sen_len: maximal text length in data
    :param X_test: test tokens
    :param Y_test: test labels
    :param is_fasttext: if vector model is fasttext
    :return:
    """
    bow_cnn_predictions = []
    original_lables_cnn_bow = []
    cnn_model.eval()

    with torch.no_grad():
        for index, row in X_test.items():
            bow_vec = vector_reprezentation.make_vector_index_map_cnn(vec_model, row, device, max_sen_len, is_fasttext)
            probs = cnn_model(bow_vec, False)
            _, predicted = torch.max(probs.data, 1)
            bow_cnn_predictions.append(predicted.cpu().numpy()[0])
            original_lables_cnn_bow.append(torch.tensor([Y_test[index]], dtype=torch.long).cpu().numpy()[0])
    # compare with true labels and print result
    app_output.output(classification_report(original_lables_cnn_bow, bow_cnn_predictions))
