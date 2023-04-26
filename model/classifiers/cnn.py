import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import optim

from view import util
from controller import vector_reprezentation

EMBEDDING_SIZE = 500
NUM_FILTERS = 10


class ConvolutionalNeuralNetworkClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, model_filename, device, window_sizes=(1, 2, 3, 5), padding=False,
                 trans_matrix=None, model_filename_test=None):
        super(ConvolutionalNeuralNetworkClassifier, self).__init__()

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
            nn.Conv2d(1, NUM_FILTERS, [window_size, vec_model.vector_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

        self.trans_matrix = None if trans_matrix is None else torch.from_numpy(trans_matrix).float().to(device)

    def forward(self, x, train_input):
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


def training_CNN(model, model_filename, trans_matrix, device, max_sen_len, X_train, Y_train_sentiment, is_fasttext,
                 padding=False, model_filename_test=None):
    NUM_CLASSES = 2
    VOCAB_SIZE = len(model)

    cnn_model = ConvolutionalNeuralNetworkClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES,
                                                     model_filename=model_filename, device=device, padding=padding,
                                                     trans_matrix=trans_matrix, model_filename_test=model_filename_test)
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
            target = torch.tensor([Y_train_sentiment[index]], dtype=torch.long, device=device)
            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            train_loss += loss.item()
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()

    return cnn_model


def testing_CNN(cnn_model, vec_model, device, max_sen_len, X_test, Y_test_sentiment, is_fasttext):
    bow_cnn_predictions = []
    original_lables_cnn_bow = []
    cnn_model.eval()

    with torch.no_grad():
        for index, row in X_test.items():
            bow_vec = vector_reprezentation.make_vector_index_map_cnn(vec_model, row, device, max_sen_len, is_fasttext)
            probs = cnn_model(bow_vec, False)
            _, predicted = torch.max(probs.data, 1)
            bow_cnn_predictions.append(predicted.cpu().numpy()[0])
            original_lables_cnn_bow.append(torch.tensor([Y_test_sentiment[index]], dtype=torch.long).cpu().numpy()[0])
    # compare with true labels and print result
    util.output(classification_report(original_lables_cnn_bow, bow_cnn_predictions))
