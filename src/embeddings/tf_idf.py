import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

import src.paths as paths
import math
import numpy as np
from scipy import sparse


def main_tf_idf(train_tweets_path, preprocessed_file, tf_idf_path, test_tweets_path, test_preprocessed_file, test_tfidf_path):
    """
    Contruct TF-IDF matrices for the training and test data.
    :param train_tweets_path: path to the file containing the training tweets.
    :param preprocessed_file: path to the file containing tweets that have been preprocessed for TF-IDF computation.
    :param tf_idf_path: path to the file where the TF-IDF matrix is stored.
    :param test_tweets_path: path to the file containing the test data.
    :param test_preprocessed_file: path to the file containing test data that have been preprocessed for TF-IDF computation.
    :param test_tfidf_path: path to the file where the TF-IDF matrix of the test set is stored
    """
    print("Train set: ")
    TF_IDF_prediction(train_tweets_path, preprocessed_file, tf_idf_path)
    print("Test set: ")
    TF_IDF_prediction(test_tweets_path, test_preprocessed_file, test_tfidf_path)


# Compute TF_IDF matrix
def TF_IDF_prediction(tweets_path, preprocessed_file, tf_idf_path):
    """
    Construct the TD-IDF matrix. For more details: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
    :param tweets_path: path to the file containing the tweets.
    :param preprocessed_file: path to the file containing tweets that have been preprocessed for TF-IDF computation
    :param tf_idf_path: path to the file where the TF-IDF matrix is stored.
    """
    cut_vocab = cut_vocab_array().values()
    preprocess_TFIDF(tweets_path, preprocessed_file, cut_vocab)
    print("preprocess done")
    TF_IDF(preprocessed_file, cut_vocab, tf_idf_path)
    print("tf_idf array ready")



def preprocess_TFIDF(tweets_path, preprocessed_file, cut_vocab):
    """
    Preprocess the tweets by removing the words in each tweet that are not in cut_vocab.
    :param tweets_path: path to the file containing the tweets.
    :param preprocessed_file: path to the file containing tweets that have been preprocessed for TF-IDF computation
    :param cut_vocab: set of words that are in cut_vocab.
    """

    outfile = open(preprocessed_file, "w+")
    with open(tweets_path, 'r') as f:
        for line in f:
            line = line[:-1]
            tweet = line.split()
            new_tweet = []
            for word in tweet:
                if word in cut_vocab:
                    new_tweet.append(word)
            new_tweet = ' '.join(new_tweet)
            outfile.write(new_tweet)
            outfile.write('\n')
        outfile.close()


def TF_IDF(tweets_path, cut_vocab, tf_idf_matrix_path):
    """
     Computes complete process to in order to construct the TF-IDF matrix for a given tweet sets.
    :param tweets_path: path to the file containing the tweets.
    :param cut_vocab: set of words that are in cut_vocab.
    :param tf_idf_matrix_path: path to the file where the TF-IDF matrix will be stored.
    """
    idf_pre_dic = compute_IDF_dict(tweets_path, cut_vocab)
    in_file_size = sum(1 for _ in open(tweets_path, 'r'))
    idf_dic = compute_IDF(in_file_size, idf_pre_dic)
    TF_IDF_array = compute_TFIDF(tweets_path, cut_vocab, in_file_size, idf_dic)
    np.save(tf_idf_matrix_path, sparse.csc_matrix(TF_IDF_array))


def compute_IDF_dict(tweets_path, cut_vocab):
    """
    For each word in cut_vocab, computes the number tweets that contains the word.
    :param tweets_path: path to the file containing the tweets.
    :param cut_vocab: set of word that are in cut_vocab
    :return: dictionary with words mapped to their occurrences in tweet set.
    """
    idf_dic = dict.fromkeys(cut_vocab.keys(), 0)
    with open(tweets_path) as f:
        for line in f:
            set_line = set(line.split())
            for x in set_line:
                idf_dic[x] += 1
    return idf_dic


def compute_IDF(tweets_size, idf_dic):
    """
    Compute the inverse data frequency (IDF) of each word in the given idf_dic dictionary
    The inverse data frequency is computed as idf(word) = log(N/dtf), where dtf is the number of tweets that contains
    the tweets and N is the total number of tweets.
    :param tweets_size: total number of tweets.
    :param idf_dic: dictionary of words in cut_vocab and their occurrences in tweets.
    :return: dictionary with words mapped to their IDF.
    """
    N = tweets_size
    for word, val in idf_dic.items():
        if val > 0:
            idf_dic[word] = math.log(N/float(val))
    return idf_dic


def compute_TF(tweet):
    """
    Compute the term frequency (TF) of each word in a given tweet. The term frequency is computed as tf(word) = n/N,
    where n is the number of time the word appears in the tweet and N is the total number of word.
    :param tweet: tweet
    :return: dictionary of words in tweets mapped to their TF.
    """
    tweet = tweet.split()
    tf_dic = dict.fromkeys(set(tweet), 0)
    for word in tweet:
        tf_dic[word] += 1
    return tf_dic


def compute_TFIDF(tweets_path, cut_vocab, tweets_size, idf_dic):
    """
    Compute the TF-IDF matrix for a given (preprocessed) tweet set.
    :param tweets_path: path to the file containing the tweets.
    :param cut_vocab: set of words that are in cut_vocab
    :param tweets_size: the number of tweets
    :param idf_dic: dictionary with words in cut_vocab mapped to their occurrences in tweet sets.
    :return: TF-IDF matrix.
    """
    cut_vocab_size = len(cut_vocab.keys())
    TF_IDF_array = np.zeros((tweets_size, cut_vocab_size))
    with open(tweets_path, 'r') as f:
        for i, tweet in enumerate(f, 0):
            tf_dic = compute_TF(tweet)
            tweet_len = len(tweet.split())
            for word, val in tf_dic.items():
                get_index = cut_vocab[word]
                TF_IDF_array[i][get_index] = val/tweet_len * idf_dic[word]
    return TF_IDF_array



def cut_vocab_array():
    """
    Construct a dictionary with the words in cut_vocab and their respective index.
    :return: dictionary with words in cut_vocab mappend to their index.
    """
    a = set(line.strip() for line in open(paths.STANFORD_NEW_CUT_VOCAB, 'r'))
    array = dict(map(lambda t: (t[1], t[0]), enumerate(a)))
    return array


if __name__ == '__main__':
     main_tf_idf(paths.TRAIN_UNIQUE, paths.PREPROCESSED_TFIDF, paths.TFIDF, paths.TEST_PREPROCESSED,
                 paths.TEST_PREPROCESSED_TFIDF, paths.TEST_TFIDF)
