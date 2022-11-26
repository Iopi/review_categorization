import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import time

from sklearn.metrics import classification_report
from torch import optim

import constants
import util
from preprocessing import preprocessing_methods
from vector_model import models


# Defining neural network structure
class LogisticRegressionClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # needs to be done everytime in the nn.module derived class
        super(LogisticRegressionClassifier, self).__init__()

        # Define the parameters that are needed for linear model ( Ax + b)
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec): # Defines the computation performed at every call.
        # Pass the input through the linear layer,
        # then pass that through log_softmax.

        return F.log_softmax(self.linear(bow_vec), dim=1)



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
            target = preprocessing_methods.make_target(Y_train_sentiment[index], device)

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
            original_lables.append(preprocessing_methods.make_target(Y_test_sentiment[index], device).cpu().numpy()[0])
    util.output(classification_report(original_lables, lr_nn_predictions))
    util.output("Time taken to predict: " + str(time.time() - start_time))



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

