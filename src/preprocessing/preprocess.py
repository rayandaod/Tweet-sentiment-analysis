import re
import os
import sys
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordsegment import load, segment
from autocorrect import Speller

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

import src.preprocessing.dictionaries as dictionaries
import src.paths as paths

stopwords = set(stopwords.words('english'))
spell = Speller(lang='en')

# OTHER VALUES
POS_LABEL = '+1'
NEG_LABEL = '-1'


def preprocess():
    """
    Perform preprocessing on the positive, negative, and test tweet set, then concatenate into one tweet set.
    """
    preprocess_pos()
    preprocess_neg()

    remove_indices_test()
    preprocess_test()

    # Concatenate the proprocessed versions of positive tweets and negative tweets into a new file
    concat_files([paths.POS_LABELS, paths.NEG_LABELS], paths.TRAIN)
    # Remove the tweets that appear >= 2 times and separate label from tweet.
    remove_both_duplicate_tweets(paths.TRAIN, paths.TRAIN_UNIQUE, paths.TRAIN_CONCAT_LABEL_UNIQUE)


def preprocess_pos():
    """
    Perform preprocessing by removing duplicate tweets, preprocessings spaces, hashtags, contracted words, smileys
    and numbers on positive tweet set.
    """
    print('Pre-processing positive tweets...')

    in_filename = remove_duplicate_tweets(paths.POS, paths.POS_UNIQUE)
    in_filename = spaces(in_filename, paths.POS_SPACES)
    in_filename = hashtags(in_filename, paths.POS_HASHTAGS)
    in_filename = contractions(in_filename, paths.POS_CONTRACT)
    in_filename = smileys(in_filename, paths.POS_SMILEYS)
    in_filename = numbers(in_filename, paths.POS_PREPROCESSED)
    in_filename = add_label(in_filename, paths.POS_LABELS, POS_LABEL)


def preprocess_neg():
    """
    Perform preprocessing by removing duplicate tweets, preprocessings spaces, hashtags, contracted words, smileys
    and numbers on negative tweet set.
    """
    print('Pre-processing negative tweets...')
    in_filename = remove_duplicate_tweets(paths.NEG, paths.NEG_UNIQUE)
    in_filename = spaces(in_filename, paths.NEG_SPACES)
    in_filename = hashtags(in_filename, paths.NEG_HASHTAGS)
    in_filename = contractions(in_filename, paths.NEG_CONTRACT)
    in_filename = smileys(in_filename, paths.NEG_SMILEYS)
    in_filename = numbers(in_filename, paths.NEG_PREPROCESSED)
    in_filename = add_label(in_filename, paths.NEG_LABELS, NEG_LABEL)


def preprocess_test():
    """
    Perform preprocessing by removing duplicate tweets, preprocessings spaces, hashtags, contracted words, smileys
    and numbers on test set.
    """
    print('Pre-processing the test set...')
    in_filename = spaces(paths.TEST_WITHOUT_INDICES, paths.TEST_SPACES)
    in_filename = hashtags(in_filename, paths.TEST_HASHTAGS)
    in_filename = contractions(in_filename, paths.TEST_CONTRACT)
    in_filename = smileys(in_filename, paths.TEST_SMILEYS)
    in_filename = numbers(in_filename, paths.TEST_PREPROCESSED)


def remove_indices_test():
    """
    Remove the index label from the tweet in test set.
    """
    test_file = open(paths.TEST, 'r')
    test_file_without_i = open(paths.TEST_WITHOUT_INDICES, 'w')
    tweets_with_indices = [tweet for tweet in test_file]
    for i in range(len(tweets_with_indices)):
        size_to_remove = len(str(i+1))+1
        test_file_without_i.write(tweets_with_indices[i][size_to_remove:])
    test_file.close()
    test_file_without_i.close()


def remove_duplicate_tweets(tweets_path, out_file_path):
    """
    Remove duplicated tweets given
    :param tweets_path: path to the file that contains tweets with duplicates
    :param out_file_path: path to the file that contains tweets without duplicates.
    :return: path to the file for the set of tweets without duplicates.
    """
    print('\tRemoving duplicate tweets...')
    lines_seen = set()
    outfile = open(out_file_path, "w")
    for line in open(out_file_path, "r"):
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    print('\t\tRemove duplicates ok.')
    return out_file_path


def remove_both_duplicate_tweets(tweets_path, out_filename, out_label_filename):
    """
    Remove tweets that are in positive tweet set and negative tweet set.
    :param tweets_path: path to the file that contains positive and negative tweets
    :param out_filename: path to the file that contains tweets without duplicates
    :param out_label_filename: path to the file that contains labels to corresponding tweets in out_filename.
    """
    print('Removing both duplicates...')
    line_to_occ = {}

    # Populate the dictionary with tweets and occurences
    for line in open(tweets_path, "r"):
        tweet = line[2:]
        label = line[:2]
        if tweet in line_to_occ:
            t = list(line_to_occ[tweet])
            t[0] += 1
            t = tuple(t)
            line_to_occ[tweet] = t
        else:
            line_to_occ[tweet] = (1, label)

    # Write the remaining tweets in the output file
    outfile = open(out_filename, "w")
    out_label_file = open(out_label_filename, 'w')
    for tweet in line_to_occ.keys():
        if line_to_occ[tweet][0] < 2:
            outfile.write(tweet)
            out_label_file.write((line_to_occ[tweet][1]))
            out_label_file.write('\n')

    outfile.close()
    out_label_file.close()
    print('\tRemove both ok.')


def spaces(tweets_path, out_filename):
    """
    Preprocess spaces by replacing multiple spaces into single one.
    :param tweets_path: path to the file that contains tweets.
    :param out_filename: path to the file that contains tweets with processed spaces.
    :return: out_filename: path to the processed file.
    """
    print('\tHandling spaces...')
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        outfile.write(re.sub(' +', ' ', tweet))
    outfile.close()
    print('\t\tSpaces ok.')
    return out_filename


def hashtags(tweets_path, out_filename):
    """
    Segment expression followed by hashtags.
    :param tweets_path: path to the file that contains tweets.
    :param out_filename: path to the file that contains hashtag expressions preprocessed.
    :return: path to the file that contains hashtag expressions preprocessed.
    """
    print('\tHandling hashtags...')
    load()
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        new_tweet = []
        list_of_words = tweet.split(' ')
        for i in range(len(list_of_words)):
            word = list_of_words[i]
            if word[0] == '#':
                for w in segment(word[1:]):
                    new_tweet.append(w)
                if i == len(list_of_words) - 1:
                    new_tweet.append('\n')
            else:
                new_tweet.append(word)
        tweet_str = []
        for i in range(len(new_tweet)):
            tweet_str.append(str(new_tweet[i]))
            if i != len(new_tweet) - 1:
                tweet_str.append(' ')
        outfile.write(''.join(tweet_str))
    outfile.close()
    print('\t\tHashtags ok.')
    return out_filename


def autocorrect(tweets_path, out_filename):
    """
    Autocorrect words in the given tweet set
    :param tweets_path: path to the file that contains the tweet set
    :param out_filename: path to the file that contains tweets with autcorrected words.
    :return: path to the file that contains tweets with autcorrected words.
    """
    print('\tAuto-correcting tweets...')
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        outfile.write(' '.join([spell(w) for w in tweet.split()]))
    outfile.close()
    print('\t\tAuto-correct ok.')
    return out_filename


def contractions(tweets_path, out_filename):
    """
    Handle contracted expression by replacing them into their expanded version.
    :param tweets_path: path to the file that contains the tweet set
    :param out_filename: path to the file that contains tweets with expanded contractions.
    :return: path to the file that contains tweets with expanded contractions.
    """
    print('\tHandling contractions...')
    outfile = open(out_filename, "w")
    contractions = dictionaries.load_dict_contractions()
    for tweet in open(tweets_path, "r"):
        tweet_list = tweet.split()
        tweet_list_new = []
        for word in tweet_list:
            if word in contractions.keys():
                tweet_list_new.append(contractions[word])
            else:
                tweet_list_new.append(word)
        outfile.write(' '.join(tweet_list_new))
        outfile.write('\n')
    outfile.close()
    print('\t\tContractions ok.')
    return out_filename


def smileys(tweets_path, out_filename):
    """
    Change any smiley into "smiley".
    :param tweets_path: path to the file that contains the tweet set with smileys.
    :param out_filename:
    :return:
    """
    print('\tHandling smileys...')
    outfile = open(out_filename, "w")
    smileys = dictionaries.load_dict_smileys()
    for tweet in open(tweets_path, "r"):
        tweet_list = tweet.split()
        tweet_list_new = []
        for word in tweet_list:
            if word in smileys.keys():
                tweet_list_new.append(smileys[word])
            else:
                tweet_list_new.append(word)
        outfile.write(' '.join(tweet_list_new))
        outfile.write('\n')
    outfile.close()
    print('\t\tSmileys ok.')
    return out_filename


def numbers(tweets_path, out_filename):
    """
    Change any numbers into <number>
    :param tweets_path: path to the file that contains the tweet set.
    :param out_filename: path to the file with any numbers replaced by <number>
    :return: path to the file with tweets where any numbers replaced by <number>
    """
    print('\tHandling numbers...')
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        outfile.write(re.sub('[-+]?\d*\.\d+|\d+', '<number>', tweet))
    outfile.close()
    print('\t\tNumbers ok.')
    return out_filename


def remove_hooks(tweets_path, out_filename):
    """
    Remove hooks.
    :param tweets_path: path to the file that contains the tweet set.
    :param out_filename: path to the file with tweets where hooks are removed.
    :return:
    """
    print('\tRemoving hooks...')
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        outfile.write(re.sub(' *<.*?> *', '', tweet))
    outfile.close()
    print('\t\tHooks ok.')
    return out_filename


def punctuation(tweets_path, out_filename):
    """
    Remove any punctuations such as !,?,-,.,...
    :param tweets_path: path to the file that contains the tweet set.
    :param out_filename: path to the file that contains tweets without punctuation.
    :return: path to the file that contains tweets without punctuation.
    """
    print('\tHandling punctuation...')
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        tweet_blob = TextBlob(tweet)
        outfile.write(' '.join(tweet_blob.words))
        outfile.write('\n')
    outfile.close()
    print('\t\tPunctuation ok.')
    return out_filename


# TODO: CHANGE THIS, stopwords contain negative words
def stopw(tweets_path, out_filename):
    """
    Remove words that are considered as stopwords by NLTK library.
    :param tweets_path: path to the file that contains the tweet set.
    :param out_filename: path to the file that contains the tweet set without stopwords.
    :return: path to the file that contains the tweet set without stopwords.
    """
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        tweet_list = tweet.split()
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords]
        outfile.write(' '.join(clean_mess))
        outfile.write('\n')
    outfile.close()
    print('\t\tStopwords ok.')
    return out_filename


def normalization(tweets_path, out_filename):
    """
    Normalise given tweet set.
    :param tweets_path: path to the file that contains tweet set.
    :param out_filename: path to the file that contains normalized tweet set.
    :return:
    """
    outfile = open(out_filename, "w")
    for tweet in open(tweets_path, "r"):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        tweet_list = tweet.split()
        for word in tweet_list:
            normalized_text = lem.lemmatize(word, 'v')
            normalized_tweet.append(normalized_text)
        outfile.write(' '.join(normalized_tweet))
        outfile.write('\n')
    outfile.close()
    print('\t\tNormalization ok.')
    return out_filename


def add_label(tweets_path, out_filename, label_value):
    """
    Add label value next to each tweet in given tweet set.
    :param in_filename: path to the file that contains the tweet set
    :param out_filename: path to the file that contains the tweets set to which a label is added.
    :param label_value: Corresponding label value the to tweet set.
    :return: path to the file that contains the tweets set to which a label is added.
    """
    outfile = open(out_filename, 'w')
    for line in open(tweets_path, 'r'):
        outfile.write(label_value)
        outfile.write(line)
    outfile.close()
    return out_filename


def concat_files(in_filenames, out_filename):
    """
    Concat both positive tweet set and negative tweet set.
    :param in_filenames:
    :param out_filename:
    :return:
    """
    print('Concatenating positive and negative files...')
    with open(out_filename, 'w') as outfile:
        for filename in in_filenames:
            with open(filename) as infile:
                for line in infile:
                    outfile.write(line)
    print('\tConcatenation ok.')


if __name__ == '__main__':
    preprocess()
