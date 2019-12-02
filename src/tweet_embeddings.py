import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import src.paths as paths
import numpy as np


def tweet_embed():
    embed(paths.TRAIN_UNIQUE, paths.TWEET_EMBEDDINGS, paths.EMBEDDINGS)


def cut_vocab_array():
    array = []
    for word in open(paths.CUT_VOCAB, 'r'):
        array.append(word[:-1])
    return array


# Form tweet embedding. For each word in tweet add up the corresponding vector.
def embed(in_file, output_file, embedding_file):
    cut_vocab = cut_vocab_array()
    embeddings = np.load(embedding_file)
    total_tweet_number = sum(1 for _ in open(in_file, 'r'))
    tweet_embedding_matrix = np.zeros((total_tweet_number, embeddings.shape[1]))
    print("Creating the tweet emmbeddings...")
    with open(in_file, 'r') as f:
        for i, tweet in enumerate(f, 0):
            tweet_embedding = np.zeros((1, embeddings.shape[1]))
            split_tweet = tweet.split()
            for word in split_tweet:
                if word in cut_vocab:
                    index = cut_vocab.index(word)
                    tweet_embedding += embeddings[index]
            if len(split_tweet) != 0:
                tweet_embedding = tweet_embedding/len(split_tweet)
            tweet_embedding_matrix[i] = tweet_embedding

    print("Tweet embeddings ok.")
    np.save(output_file, tweet_embedding_matrix)
    return tweet_embedding_matrix


if __name__ == '__main__':
    tweet_embed()
