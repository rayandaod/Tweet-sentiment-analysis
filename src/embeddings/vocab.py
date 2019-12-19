from nltk.tokenize import TweetTokenizer
from collections import Counter, OrderedDict
import pickle
import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

import src.paths as paths
import src.params as params


def vocab(train_tweets_path, vocab_path, cut_vocab_path, cut_vocab_n):
    """
    Build the vocabulary of our corpus, and only keep the words appearing more than cut_vocab_n times.

    :param train_tweets_path: the path to the preprocessed training set of tweets
    :param vocab_path: the path to the future vocabulary file
    :param cut_vocab_path: the path to the future restricted (as mentionned above) vocabulary file
    :param cut_vocab_n: the threshold below which the words of the vocabulary are judged irrelevant
    """
    # Build the vocabulary
    counter = build_vocab(train_tweets_path, vocab_path)
    # Only keep the tokens that appear more than n times
    cut_vocab(counter, cut_vocab_path, cut_vocab_n)
    # Pickle the vocabulary
    pickle_vocab()


def build_vocab(train_tweets_path, vocab_path, reduce_len=False):
    """
    Build the vocabulary of words/tokens contained in our set of tweets.

    :param train_tweets_path: the path to the train set of tweets
    :param vocab_path: the path to the future vocabulary file
    :param reduce_len: rather we want to reduce the length of the words that have repetitions of characters or not
    :return: a Counter variable in which there is the number of appearance for each token in our tweet set
    """
    print('Building the vocabulary...')
    # Retrieve tokens in the input file
    tknzr = TweetTokenizer(reduce_len=reduce_len)
    with open(train_tweets_path, 'r') as f:
        data = f.read()
        tokens = tknzr.tokenize(data)

    tokens.sort()
    counter = Counter(tokens)

    # Write each token in the output file
    output_file = open(vocab_path, 'w+')
    for key, value in counter.items():
        output_file.write(str(value) + ',' + key + '\n')
    output_file.close()

    print('\tVocabulary ok.')
    return counter


def cut_vocab(counter: Counter, cut_vocab_path, cut_vocab_n):
    """
    Keep the tokens appearing more than n-1 (n, n+1, ...)

    :param counter: the Counter variable storing the number of appearance for each token in our tweet set
    :param cut_vocab_path: the path to the future restricted vocabulary
    :param cut_vocab_n: the threshold below which the words of the vocabulary are judged irrelevant
    :return: the list of tokens stored in in our restricted vocabulary
    """
    print('Keeping only the words appearing {} times...'.format(cut_vocab_n))
    cut_counter = OrderedDict(counter.most_common())
    cut_counter = {x: cut_counter[x] for x in cut_counter if cut_counter[x] >= cut_vocab_n}
    outfile = open(cut_vocab_path, 'w+')
    for key in cut_counter.keys():
        outfile.write(key+'\n')
    outfile.close()
    print('\t{} words in cut_vocab.'.format(len(cut_counter.keys())))
    print('\tcut_vocab ok.')
    return counter.keys()


def pickle_vocab():
    """
    Store the restricted vocabulary previously created in a pickle file for easier access.

    """
    print('Pickling cut_vocab...')
    vocab = dict()
    with open(paths.CUT_VOCAB) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(paths.VOCAB_PICKLE, 'wb+') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print('\tcut_vocab pickled.')


if __name__ == '__main__':
    vocab(paths.TRAIN_UNIQUE, paths.VOCAB, paths.CUT_VOCAB, params.CUT_VOCAB_N)
