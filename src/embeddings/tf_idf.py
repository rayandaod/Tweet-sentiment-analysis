import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

import src.paths as paths
import math
import numpy as np
from scipy import sparse


def main_tf_idf(in_file, preprocessed_file, tf_idf_path, in_test_file, preprocessed_test_file, test_tfidf_path):
    print("Train set: ")
    TF_IDF_prediction(in_file, preprocessed_file, tf_idf_path)
    #print("Test set: ")
    #TF_IDF_prediction(in_test_file, preprocessed_test_file)


# Compute TF_IDF matrix
def TF_IDF_prediction(in_file, preprocessed_file,tf_idf_path):
    cut_vocab = cut_vocab_array()
    preprocess_TFIDF(in_file, preprocessed_file, cut_vocab)
    print("preprocess done")
    TF_IDF(preprocessed_file, cut_vocab,tf_idf_path)
    print("tf_idf array ready")


# filter the tweets with only words in cut_vocab
def preprocess_TFIDF(in_file, out_filename, cut_vocab):
    outfile = open(out_filename, "w")
    with open(in_file, 'r') as f:
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


def TF_IDF(in_file, cut_vocab, tf_idf_matrix_path):
    idf_pre_dic = compute_IDF_dict(in_file, cut_vocab)
    in_file_size = sum(1 for _ in open(in_file, 'r'))
    idf_dic = compute_IDF(in_file_size, idf_pre_dic)
    TF_IDF_array = compute_TFIDF(in_file, cut_vocab, in_file_size, idf_dic)
    np.save(tf_idf_matrix_path, sparse.csc_matrix(TF_IDF_array))



def compute_IDF_dict(in_file, cut_vocab):
    idf_dic = dict.fromkeys(cut_vocab.keys(), 0)
    with open(in_file) as f:
        for line in f:
            set_line = set(line.split())
            for x in set_line:
                idf_dic[x] += 1
    return idf_dic


def compute_IDF(tweets_size, idf_dic):
    N = tweets_size
    for word, val in idf_dic.items():
        if val > 0:
            idf_dic[word] = math.log(N/float(val))
    return idf_dic


def compute_TF(tweet):
    tweet = tweet.split()
    tf_dic = dict.fromkeys(set(tweet), 0)
    for word in tweet:
        tf_dic[word] += 1
    return tf_dic


def compute_TFIDF(in_file, cut_vocab, tweets_size, idf_dic):
    cut_vocab_size = len(cut_vocab.keys())
    TF_IDF_array = np.zeros((tweets_size, cut_vocab_size))
    with open(in_file,'r') as f:
        for i, tweet in enumerate(f, 0):
            tf_dic = compute_TF(tweet)
            tweet_len = len(tweet.split())
            for word, val in tf_dic.items():
                get_index = cut_vocab[word]
                TF_IDF_array[i][get_index] = val/tweet_len * idf_dic[word]
    return TF_IDF_array


def cut_vocab_array():
    a = set(line.strip() for line in open(paths.STANFORD_NEW_CUT_VOCAB, 'r'))
    array = dict(map(lambda t: (t[1], t[0]), enumerate(a)))
    return array


if __name__ == '__main__':
     main_tf_idf(paths.TRAIN_UNIQUE, paths.PREPROCESSED_TFIDF, paths.TFIDF, paths.TEST_PREPROCESSED, paths.TEST_PREPROCESSED_TFIDF, paths.TEST_TFIDF)