from nltk.tokenize import TweetTokenizer
from collections import Counter, OrderedDict
import pickle
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.paths as paths
import src.params as params


def vocab():
    # Build the vocabulary
    counter = build_vocab(paths.TRAIN_UNIQUE, paths.VOCAB)
    # Only keep the tokens that appear more than n times
    vocabulary = cut_vocab(counter, paths.CUT_VOCAB, n=params.CUT_VOCAB_N)
    # Pickle the vocabulary
    pickle_vocab()


def build_vocab(in_filename, out_filename, reduce_len=False):
    # Retrieve tokens in the input file
    tknzr = TweetTokenizer(reduce_len=reduce_len)
    with open(in_filename, 'r') as f:
        data = f.read()
        tokens = tknzr.tokenize(data)

    tokens.sort()
    counter = Counter(tokens)

    # Write each token in the output file
    output_file = open(out_filename, 'w')
    for key, value in counter.items():
        output_file.write(str(value) + ',' + key + '\n')
    output_file.close()

    return counter


def cut_vocab(counter: Counter, out_filename, n=5):
    cut_counter = OrderedDict(counter.most_common())
    cut_counter = {x: cut_counter[x] for x in cut_counter if cut_counter[x] >= n}
    outfile = open(out_filename, 'w')
    for key in cut_counter.keys():
        outfile.write(key+'\n')
    outfile.close()
    return counter.keys()


def pickle_vocab():
    vocab = dict()
    with open(paths.CUT_VOCAB) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(paths.VOCAB_PICKLE, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    vocab()
