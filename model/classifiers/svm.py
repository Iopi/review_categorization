import pandas as pd
from sklearn.svm import LinearSVC


def training_Linear_SVM(Y_train, model_filename):
    """
     Train Support Vector Machines classifier
     :param Y_train: train labels
     :param model_filename: model filename
     :return: Support Vector Machines classifier
     """
    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename, sep=';')
    # Initialize the model
    clf_svm = LinearSVC(random_state=0)
    # Fit the model
    clf_svm.fit(vectors_model, Y_train)
    return clf_svm

