import pandas as pd
from sklearn.linear_model import LogisticRegression

def training_Logistic_Regression(Y_train, model_filename):
    """
     Train Logistic Regression classifier
     :param Y_train: train labels
     :param model_filename: model filename
     :return: Logistic Regression classifier
     """
    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename, sep=';')
    # Initialize the model
    clf = LogisticRegression(random_state=0)
    # Fit the model
    clf.fit(vectors_model, Y_train)
    return clf

