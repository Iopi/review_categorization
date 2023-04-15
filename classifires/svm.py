import pandas as pd
from sklearn.svm import LinearSVC


def training_Linear_SVM(Y_train_sentiment, model_filename):
    # start_time = time.time()

    # Read the TFIDF/BOW vectors
    vectors_model = pd.read_csv(model_filename, sep=';')

    # Initialize the model
    clf_svm = LinearSVC(random_state=0)

    # Fit the model
    clf_svm.fit(vectors_model, Y_train_sentiment)
    # print("Time to taken to fit the TF-IDF as input for classifier: " + str(time.time() - start_time))
    return clf_svm

