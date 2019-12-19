import os
import sys
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow.keras as keras
from tensorflow_core.python.training import learning_rate_decay

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

    if params.SUBMISSION:
        helper.write_predictions_in_csv(label_predictions, predictions_file_path)


def logistic_regression(tweet_embeddings, labels, test_embeddings):
    """
    Train a logistic regression model on the train set and predict the labels of the test set.
    This logistic regression function finds its best parameters thanks to a K-fold-cross-validation, the number of
    folds being specified in the params file as CV_FOLDS.

    :param tweet_embeddings: the embedding vectors of the train set
    :param labels: the labels of the tweets of the train set
    :param test_embeddings: the embedding vectors of the test set
    :return: the labels prediction for the tweets in the test set
    """
    clf = LogisticRegressionCV(cv=params.CV_FOLDS)
    clf.fit(tweet_embeddings, labels)
    print(clf.score(tweet_embeddings, labels))
    return clf.predict(test_embeddings)


def neural_network(tweet_embeddings, labels, test_embeddings):
    """
    Train a simple neural network

    :param tweet_embeddings:
    :param labels:
    :param test_embeddings:
    :return:
    """

    if params.SUBMISSION:
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, input_dim=tweet_embeddings.shape[1], activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(tweet_embeddings, np.asarray(helper.transform_labels(labels)), epochs=params.NN_N_EPOCHS,
                  batch_size=params.NN_BATCH_SIZE, verbose=params.NN_VERBOSE)
        return helper.inv_transform_labels(np.round(model.predict(test_embeddings)))
    else:
        for train_index, test_index in StratifiedKFold(params.CV_FOLDS, shuffle=True).split(tweet_embeddings, labels):
            X_train, X_test = [tweet_embeddings[x] for x in train_index], [tweet_embeddings[x] for x in test_index]
            y_train, y_test = [labels[x] for x in train_index], [labels[x] for x in test_index]

            model = keras.Sequential()
            model.add(keras.layers.Dense(256, input_dim=tweet_embeddings.shape[1], activation='relu'))
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=params.NN_PATIENCE)
            model.fit(np.asarray(X_train), np.asarray(helper.transform_labels(y_train)), epochs=params.NN_N_EPOCHS,
                      batch_size=params.NN_BATCH_SIZE, verbose=params.NN_VERBOSE,
                      validation_data=(np.asarray(X_test), np.asarray(helper.transform_labels(y_test))),
                      callbacks=[es])


if __name__ == '__main__':
    predict(paths.TEST_EMBEDDINGS, paths.TWEET_EMBEDDINGS, paths.TRAIN_CONCAT_LABEL_UNIQUE, paths.LABEL_PREDICTIONS)
