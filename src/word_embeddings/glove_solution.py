#!/usr/bin/env python3
# from scipy.sparse import *
import numpy as np
import pickle
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.paths as paths
import src.params as params


def glove_GD(cooc_pkl, nmax, K, eta, alpha, epochs, embeddings):
    print("loading cooccurrence matrix")
    with open(cooc_pkl, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], K))
    ys = np.random.normal(size=(cooc.shape[1], K))

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
    np.save(embeddings, xs)


if __name__ == '__main__':
    glove_GD(paths.COOC_PICKLE, params.GLOVE_NMAX, params.GLOVE_K, params.GLOVE_ETA, params.GLOVE_ALPHA,
             params.GLOVE_N_EPOCHS, paths.EMBEDDINGS)
