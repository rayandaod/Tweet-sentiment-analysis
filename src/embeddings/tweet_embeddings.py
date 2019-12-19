import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import src.paths as paths
import numpy as np


def tweet_embed(train_tweets_path, train_tweets_embeddings_path, test_tweets_path, test_tweets_embeddings_path,
                cut_vocab_embeddings_path):
    """
    Add the embedding vectors of each word of a tweet together to form an embedding vector for each tweet.

    :param train_tweets_path: the path to the train set of tweets
    :param train_tweets_embeddings_path: the path to the embeddings of the tweets of our train set
    :param test_tweets_path: the path to the test set of tweets
    :param test_tweets_embeddings_path: the path to the embeddings of the tweets of our test set
    :param cut_vocab_embeddings_path: the path to the embeddings of our current vocabulary
    """
    print('Train set')
    embed(train_tweets_path, train_tweets_embeddings_path, cut_vocab_embeddings_path)

    print('Test set')
    embed(test_tweets_path, test_tweets_embeddings_path, cut_vocab_embeddings_path)


def cut_vocab_array():
    """
    Generate a list of words contained in our vocabulary.
    :return: the list of words contained in our vocabulary
    """
    cut_vocab_list = []
    for word in open(paths.STANFORD_NEW_CUT_VOCAB, 'r'):
        cut_vocab_list.append(word[:-1])
    return cut_vocab_list


# Form tweet embedding. For each word in tweet add up the corresponding vector.
def embed(train_tweets_path, train_tweets_embeddings_path, cut_vocab_embeddings_path):
    """
    Form the tweet embeddings. For each word in each tweet, add up the corresponding word embedding vectors together.
    
    :param train_tweets_path: the path to the train set of tweets
    :param train_tweets_embeddings_path: the path to the embeddings of the tweets of our train set
    :param cut_vocab_embeddings_path: the path to the embeddings of our current vocabulary
    :return: the tweet embedding matrix, i.e the matrix with shape (num_tweets, embedding_dimension)
    """
    cut_vocab = cut_vocab_array()
    embeddings = np.load(cut_vocab_embeddings_path)
    total_tweet_number = sum(1 for _ in open(train_tweets_path, 'r'))
    tweet_embedding_matrix = np.zeros((total_tweet_number, embeddings.shape[1]))
    print("\tCreating the tweet embeddings...")
    with open(train_tweets_path, 'r') as f:
        for i, tweet in enumerate(f, 0):
            tweet_embedding = np.zeros((1, embeddings.shape[1]))
            split_tweet = tweet.split()
            for word in split_tweet:
                if word in cut_vocab:
                    index = cut_vocab.index(word)  # TODO: make it faster and more furious
                    tweet_embedding += embeddings[index]
            if len(split_tweet) != 0:
                tweet_embedding = tweet_embedding/len(split_tweet)
            tweet_embedding_matrix[i] = tweet_embedding
            print('{}/{}'.format(i, total_tweet_number))

    print("\t\tTweet embeddings ok.")
    np.save(train_tweets_embeddings_path, tweet_embedding_matrix)
    return tweet_embedding_matrix


if __name__ == '__main__':
    tweet_embed(paths.TRAIN_UNIQUE, paths.TWEET_EMBEDDINGS, paths.TEST_PREPROCESSED, paths.TEST_EMBEDDINGS,
                paths.STANFORD_EMBEDDINGS_CUT_VOCAB)
