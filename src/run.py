import os
import sys
import subprocess
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.preprocessing.preprocess as preprocess
import src.embeddings.vocab as vocab
import src.embeddings.stanford_word_embeddings as stanford
import src.embeddings.tweet_embeddings as tweet_embeddings
import src.prediction.predict as predict
import src.prediction.better_predict as better_predict
import src.paths as paths
import src.params as params

if __name__ == '__main__':
    # Clear the outputs of eventual previous running
    subprocess.call(['./clear_outputs.sh'])

    # Preprocess the train and test sets
    preprocess.preprocess()

    # Build the vocabulary and only keep the words repeating CUT_VOCAB_N times
    vocab.vocab(paths.TRAIN_UNIQUE, paths.VOCAB, paths.CUT_VOCAB, params.CUT_VOCAB_N)

    # Convert the txt file from Stanford into a pickle to easily access to it
    stanford.stanford_to_pkl(paths.STANFORD_EMBEDDINGS_TXT, paths.STANFORD_EMBEDDINGS_PICKLE)

    # Store the pre-trained embeddings of the words of our vocabulary
    # We throw away the words that are not in Stanford vocabulary
    stanford.stanford_only_cut_vocab(paths.STANFORD_EMBEDDINGS_PICKLE, paths.STANFORD_EMBEDDINGS_CUT_VOCAB,
                                     paths.CUT_VOCAB, paths.STANFORD_NEW_CUT_VOCAB)

    # Build the tweet embeddings by adding the embeddings of each of its word
    tweet_embeddings.tweet_embed(paths.TRAIN_UNIQUE, paths.TWEET_EMBEDDINGS, paths.TEST_PREPROCESSED,
                                 paths.TEST_EMBEDDINGS, paths.STANFORD_EMBEDDINGS_CUT_VOCAB)

    # Train the model and classify the test set according to the model we found
    # predict.predict(paths.TEST_EMBEDDINGS, paths.TWEET_EMBEDDINGS, paths.TRAIN_CONCAT_LABEL_UNIQUE, paths.LABEL_PREDICTIONS)
    better_predict.better_predict()
