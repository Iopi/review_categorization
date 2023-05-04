import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def training_Decision_Tree(Y_train, model_filename):
    """
    Train Decision Tree classifier
    :param Y_train: train labels
    :param model_filename: model filename
    :return: Decision Tree classifier
    """
    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename, sep=';')
    # Initialize the model
    clf_decision = DecisionTreeClassifier(random_state=2)
    # Fit the model
    clf_decision.fit(vectors_model, Y_train)
    return clf_decision
