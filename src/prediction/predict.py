import os
import sys
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

import tensorflow.keras as keras

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

import src.paths as paths
import src.params as params
import src.prediction.predict_helper as helper


def predict(test_embeddings_path, tweet_embeddings_path, labels_path, predictions_file_path):
    """
    Run the chosen machine learning algorithm on the train set of tweets to classify the tweets of the test set.

    :param test_embeddings_path: the path to the embedding vectors of the test set
    :param tweet_embeddings_path: the path to the embedding vectors of the train set
    :param labels_path: the path to the labels of the tweets of the train set
    :param predictions_file_path: the path to the future file containing the predictions
    """
    test_embeddings = np.load(test_embeddings_path)
    tweet_embeddings = np.load(tweet_embeddings_path)
    labels = helper.labels_list(labels_path)

    label_predictions = neural_network(tweet_embeddings, labels, test_embeddings)
    # label_predictions = logistic_regression(tweet_embeddings, labels, test_embeddings)

    helper.write_predictions_in_csv(label_predictions, predictions_file_path)


def logistic_regression(tweet_embeddings, labels, test_embeddings):
    clf = LogisticRegressionCV(cv=params.CV_FOLDS)
    clf.fit(tweet_embeddings, labels)
    print(clf.score(tweet_embeddings, labels))
    return clf.predict(test_embeddings)


def neural_network(tweet_embeddings, labels, test_embeddings):
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_dim=tweet_embeddings.shape[1], activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(tweet_embeddings, helper.transform_labels(labels),
                                                        test_size=params.TEST_SIZE)

    model.fit(X_train, np.asarray(y_train), epochs=params.NN_N_EPOCHS, batch_size=params.NN_BATCH_SIZE, verbose=params.NN_VERBOSE)
    _, accuracy = model.evaluate(X_test, np.asarray(y_test), verbose=params.NN_VERBOSE)
    print('Accuracy = {}'.format(accuracy * 100))
    return helper.inv_transform_labels(np.round(model.predict(test_embeddings)))


if __name__ == '__main__':
    predict(paths.TEST_EMBEDDINGS, paths.TWEET_EMBEDDINGS, paths.TRAIN_CONCAT_LABEL_UNIQUE, paths.LABEL_PREDICTIONS)
