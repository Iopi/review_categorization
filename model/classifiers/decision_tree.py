import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def training_Decision_Tree(Y_train_sentiment, model_filename):
    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename, sep=';')
    # Initialize the model
    clf_decision = DecisionTreeClassifier(random_state=2)
    # Fit the model
    clf_decision.fit(vectors_model, Y_train_sentiment)
    return clf_decision
