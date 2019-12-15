import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')
import src.paths as paths
import numpy as np
import src.embeddings.tf_idf as tfidf


def tweet_embed(train_tweets_path, train_tweets_embeddings_path, test_tweets_path, test_tweets_embeddings_path,
                cut_vocab_embeddings_path):
    print('Train set')
    embed(train_tweets_path, train_tweets_embeddings_path, cut_vocab_embeddings_path)

    print('Test set')
    embed(test_tweets_path, test_tweets_embeddings_path, cut_vocab_embeddings_path)


# Form tweet embedding. For each word in tweet add up the corresponding vector.
def embed(in_file, output_file, embedding_file, tfidf_matrix_file, tfidf_bool = False):
    cut_vocab = tfidf.cut_vocab_array()
    embeddings = np.load(embedding_file)
    tfidf_matrix = np.load(tfidf_matrix_file)
    total_tweet_number = tfidf_matrix.shape[0]
    tweet_embedding_matrix = np.zeros((total_tweet_number, embeddings.shape[1]))
    print("\tCreating the tweet embeddings...")
    with open(in_file, 'r') as f:
        for i, tweet in enumerate(f, 0):
            tweet_embedding = np.zeros((1, embeddings.shape[1]))
            split_tweet = tweet.split()
            for word in split_tweet:
                if word in cut_vocab:
                    index = cut_vocab[word]
                    if tfidf_bool:
                        tweet_embedding += tfidf_matrix[i][index] * embeddings[index]
                    else:
                        tweet_embedding += embeddings[index]
            if len(split_tweet) != 0:
                tweet_embedding = tweet_embedding/len(split_tweet)
            tweet_embedding_matrix[i] = tweet_embedding
            print('{}/{}'.format(i, total_tweet_number))

    print("\t\tTweet embeddings ok.")
    np.save(output_file, tweet_embedding_matrix)
    return tweet_embedding_matrix


if __name__ == '__main__':
    tweet_embed(paths.TRAIN_UNIQUE, paths.TWEET_EMBEDDINGS, paths.TEST_PREPROCESSED, paths.TEST_EMBEDDINGS,
                paths.STANFORD_EMBEDDINGS_CUT_VOCAB)
