import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import gensim
import torch

class LongShortTermMemory(nn.Module):

    def __init__(self, model_filename, padding=True, dimension=128):
        super(LongShortTermMemory, self).__init__()

        vec_model = gensim.models.KeyedVectors.load(model_filename)
        weights = vec_model.wv

        # With pretrained embeddings
        if padding:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors),
                                                          padding_idx=vec_model.wv.key_to_index['pad'])
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors))
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=500,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)

        self.dense = nn.Linear(2*dimension, 1)

        # self.lstm = nn.LSTM(500, 300, 1, bidirectional=False)
        # self.dropout = nn.Dropout(0.5)
        # self.dense = nn.Linear(300, 5)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text_len = self.get_text_len(text)
        #
        # text_emb = self.embedding(text)
        #
        # packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        # packed_output, _ = self.lstm(packed_input)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #
        # out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        # out_reverse = output[:, 0, self.dimension:]
        # out_reduced = torch.cat((out_forward, out_reverse), 1)
        # text_fea = self.drop(out_reduced)
        #
        # text_fea = self.fc(text_fea)
        # text_fea = torch.squeeze(text_fea, 1)
        # text_out = torch.sigmoid(text_fea)

        # return text_out

        text_emb = self.embedding(text)
        lstm_out, _ = self.lstm(text_emb)
        lstm_out = lstm_out[:, -1, :]
        drop_out = self.dropout(lstm_out)
        output = self.dense(drop_out)
        # sigmoid_out = self.sigmoid(output)

        return output


    def get_text_len(self, text):
        arr = []
        for x in range(len(text)):
            counter = 0
            for y in range(len(text[x])):
                if text[x][y] != 1:
                    counter += 1
            arr.append(counter)

        return torch.tensor(arr)