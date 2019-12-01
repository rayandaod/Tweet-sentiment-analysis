from nltk.tokenize import TweetTokenizer
from collections import Counter, OrderedDict
from pathlib import Path
import pickle
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.preprocess as prep
import src.params as params

TRAIN_CONCAT = Path(BASE_PATH + "/data/preprocessed/train.txt")
TRAIN_CONCAT_UNIQUE = Path(BASE_PATH + "/data/preprocessed/train_unique.txt")


def vocab():
    # Concatenate the proprocessed versions of positive tweets and negative tweets into a new file
    concat_files([params.POS_PREPROCESSED, params.NEG_PREPROCESSED], TRAIN_CONCAT)
    # Remove the tweet that appear >= 2 times
    prep.remove_both_duplicate_tweets(TRAIN_CONCAT, TRAIN_CONCAT_UNIQUE)
    # Build the vocabulary
    counter = build_vocab(TRAIN_CONCAT_UNIQUE, params.VOCAB)
    # Only keep the tokens that appear more than n times
    vocabulary = cut_vocab(counter, params.CUT_VOCAB, n=params.CUT_VOCAB_N)
    # Pickle the vocabulary
    pickle_vocab()


def concat_files(in_filenames, out_filename):
    with open(out_filename, 'w') as outfile:
        for filename in in_filenames:
            with open(filename) as infile:
                for line in infile:
                    outfile.write(line)


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
    id = 1
    for key in cut_counter.keys():
        outfile.write(str(id) + ',' + key+'\n')
        id += 1
    outfile.close()
    return counter.keys()


def pickle_vocab():
    vocab = dict()
    with open(params.CUT_VOCAB) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(params.VOCAB_PICKLE, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    vocab()
