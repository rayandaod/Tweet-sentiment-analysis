import src.params as params
import src.preprocessing as prep
import vocab

if __name__ == "__main__":
    # Remove duplicate tweets in each of the two training datasets (positive, and negative)
    prep.remove_duplicate_tweets(params.POS, params.POS_UNIQUE)
    prep.remove_duplicate_tweets(params.NEG, params.NEG_UNIQUE)

    # Pre-processing stuff
    prep.numbers(params.POS_UNIQUE, 'data/preprocessed/train_pos_numbers.txt')
    prep.remove_hooks('data/preprocessed/train_pos_numbers.txt', 'data/preprocessed/train_pos_hooks.txt')
    prep.punctuation('data/preprocessed/train_pos_hooks.txt', 'data/preprocessed/train_pos_punct.txt')

    # # Concatenate both files (containing unique positive tweets, and unique negative tweets) into one
    # vocab.concat_files([params.POS_UNIQUE, params.NEG_UNIQUE])
    #
    # # Remove the tweets that appear >= 2 times (which means they were both in the positive, and negative dataset)
    # prep.remove_both_duplicate_tweets(params.TRAIN_CONCAT, params.TRAIN_CONCAT_UNIQUE)
    #
    # # Build the vocabulary
    # vocab.build_vocab(params.TRAIN_CONCAT_UNIQUE, params.VOCAB)
