import os
import sys
import numpy as np
import csv
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tk = Tokenizer()


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.paths as paths
import src.params as params
import src.preprocess as prep
import src.tweet_embeddings as t_embed


from keras.models import Sequential
from keras.layers import Dense


def predict():
    remove_indices_test()
    preprocess_test()
    test_embeddings = t_embed.embed(paths.TEST_PREPROCESSED, paths.TEST_EMBEDDINGS, paths.EMBEDDINGS)
    tweet_embeddings = np.load(paths.TWEET_EMBEDDINGS)

    with open(paths.TRAIN_CONCAT_LABEL_UNIQUE, 'r') as train_labels_file:
        labels = [int(label[:-1]) for label in train_labels_file]


    model = Sequential()
    model.add(Dense(12, input_dim=tweet_embeddings.shape[1], activation="relu"))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(tweet_embeddings, labels, test_size=params.TEST_SIZE)
    y_train_swapped = [1 if x== -1 else -1 for x in y_train]
    y_test_swapped = [1 if x==-1 else -1 for x in y_test]


    print("Fitting has started.")
    model.fit(X_train, y_train_swapped, epochs=20, batch_size=1)
    print("Predicting has started.")
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))

    #label_predictions = logistic_regression(tweet_embeddings, labels, test_embeddings)
    #write_predictions_in_csv(label_predictions)


def preprocess_test():
    print('Pre-processing the test set...')
    in_filename = prep.spaces(paths.TEST_WITHOUT_INDICES, paths.TEST_SPACES)
    in_filename = prep.hashtags(in_filename, paths.TEST_HASHTAGS)
    in_filename = prep.contractions(in_filename, paths.TEST_CONTRACT)
    in_filename = prep.smileys(in_filename, paths.TEST_SMILEYS)
    in_filename = prep.remove_hooks(in_filename, paths.TEST_PREPROCESSED)


def remove_indices_test():
    test_file = open(paths.TEST, 'r')
    test_file_without_i = open(paths.TEST_WITHOUT_INDICES, 'w')
    tweets_with_indices = [tweet for tweet in test_file]
    for i in np.arange(len(tweets_with_indices)):
        size_to_remove = len(str(i+1))+1
        test_file_without_i.write(tweets_with_indices[i][size_to_remove:])
    test_file.close()
    test_file_without_i.close()


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


def write_predictions_in_csv(label_predictions):
    labels_with_ids = ['Id,Prediction']
    for i in np.arange(len(label_predictions)):
        labels_with_ids.append(str(i + 1) + ',' + str(label_predictions[i]))

    with open(paths.LABEL_PREDICTIONS, 'w') as result_file:
        wr = csv.writer(result_file, delimiter=',')
        wr.writerows([x.split(',') for x in labels_with_ids])


if __name__ == '__main__':
    predict()
