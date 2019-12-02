import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import src.params as params
import numpy as np


def tweet_embed():
    embed(params.TRAIN_UNIQUE, params.TWEET_EMBEDDINGS, params.EMBEDDINGS)


def cut_vocab_array():
    array = []
    for word in open(params.CUT_VOCAB,'r'):
        array.append(word[:-1])
    print("cut_vocab is ready")
    return array


# Form tweet embedding. For each word in tweet add up the corresponding vector.
def embed(in_file, output_file,embedding_file):

    cut_vocab = cut_vocab_array()
    embeddings = np.load(embedding_file)
    total_tweet_number = sum(1 for _ in open(in_file, 'r'))
    tweet_embedding_matrix = np.zeros((total_tweet_number, embeddings.shape[1]))
    print("Embedding started")
    with open(in_file, 'r') as f:
        for i, tweet in enumerate(f, 0):
            split_tweet = tweet.split()
            for word in split_tweet:
                if word in cut_vocab:
                    index = cut_vocab.index(word)
                    tweet_embedding_matrix[i] += embeddings[index]
    print("Embedding finished")
    np.save(output_file, tweet_embedding_matrix)


if __name__ == '__main__':
    tweet_embed()