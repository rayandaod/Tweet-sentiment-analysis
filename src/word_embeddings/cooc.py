#!/usr/bin/env python3
from scipy.sparse import *
import pickle
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.paths as paths


def cooc(vocab_pickle, pos_tweets, neg_tweets, cooc_pickle):
    with open(vocab_pickle, 'rb') as f:
        vocab = pickle.load(f)

    data, row, col = [], [], []
    counter = 1
    for fn in [pos_tweets, neg_tweets]:
        with open(fn) as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("Summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open(cooc_pickle, 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cooc(paths.VOCAB_PICKLE, paths.POS, paths.NEG, paths.COOC_PICKLE)
