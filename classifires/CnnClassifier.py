

import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F

import constants

# import torch.optim as optim

EMBEDDING_SIZE = 500
NUM_FILTERS = 10


class ConvolutionalNeuralNetworkClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, model_filename, window_sizes=(1,2,3,5), padding=False):
        super(ConvolutionalNeuralNetworkClassifier, self).__init__()
        vec_model = gensim.models.KeyedVectors.load(model_filename)
        weights = vec_model.wv
        # With pretrained embeddings
        if padding:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=vec_model.wv.key_to_index['pad'])
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors))
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x) # [B, T, E]

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

        return probs
