import os
import sys
import numpy as np
import csv
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

import src.paths as paths
import src.params as params


def predict(test_embeddings_path, tweet_embeddings_path, labels_path):
    test_embeddings = np.load(test_embeddings_path)
    tweet_embeddings = np.load(tweet_embeddings_path)
    labels = labels_list(labels_path)

    label_predictions = neural_network(tweet_embeddings, labels, test_embeddings)
    # label_predictions = logistic_regression(tweet_embeddings, labels, test_embeddings)

    write_predictions_in_csv(label_predictions)


def logistic_regression(tweet_embeddings, labels, test_embeddings):
    clf = LogisticRegressionCV(cv=params.CV_FOLDS)
    clf.fit(tweet_embeddings, labels)
    print(clf.score(tweet_embeddings, labels))
    return clf.predict(test_embeddings)


def support_vector_machines(tweet_embeddings, labels, test_embeddings):
    X_train, X_test, y_train, y_test = train_test_split(tweet_embeddings, labels, test_size=params.TEST_SIZE)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    return clf.predict(test_embeddings)


def neural_network(tweet_embeddings, labels, test_embeddings):
    model = Sequential()
    model.add(Dense(256, input_dim=tweet_embeddings.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(tweet_embeddings, transform_labels(labels),
                                                        test_size=params.TEST_SIZE)

    print("Fitting has started.")
    model.fit(X_train, y_train, epochs=params.NN_N_EPOCHS, batch_size=params.NN_BATCH_SIZE, verbose=params.NN_VERBOSE)
    print("Predicting has started.")
    _, accuracy = model.evaluate(X_test, y_test, verbose=params.NN_VERBOSE)
    print('Accuracy = {}'.format(accuracy * 100))
    return inv_transform_labels(np.round(model.predict(test_embeddings)))


def write_predictions_in_csv(label_predictions):
    labels_with_ids = ['Id,Prediction']
    for i in np.arange(len(label_predictions)):
        labels_with_ids.append(str(i + 1) + ',' + str(label_predictions[i]))

    with open(paths.LABEL_PREDICTIONS, 'w') as result_file:
        wr = csv.writer(result_file, delimiter=',')
        wr.writerows([x.split(',') for x in labels_with_ids])


def labels_list(labels_file_path):
    with open(labels_file_path, 'r') as train_labels_file:
        return [int(label[:-1]) for label in train_labels_file]


def transform_labels(y):
    return [(x + 1) / 2 for x in y]


def inv_transform_labels(y_transformed):
    return [int(2*x-1) for x in y_transformed]


def clip_labels(labels):
    labels_clipped = []
    for y in labels:
        if y < 0:
            labels_clipped.append(-1)
        else:
            labels_clipped.append(1)
    return labels_clipped


if __name__ == '__main__':
    predict(paths.TEST_EMBEDDINGS, paths.TWEET_EMBEDDINGS, paths.TRAIN_CONCAT_LABEL_UNIQUE)
