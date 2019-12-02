from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordsegment import load, segment
from autocorrect import Speller
from pathlib import Path
import re

import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import src.dictionaries as dictionaries
import src.params as params

stopwords = set(stopwords.words('english'))
spell = Speller(lang='en')

pos_preprocessed_path = Path(BASE_PATH + '/data/preprocessed/pos')
neg_preprocessed_path = Path(BASE_PATH + '/data/preprocessed/neg')

POS_UNIQUE = pos_preprocessed_path / "train_pos_unique.txt"
POS_FULL = "../data/train_pos_full.txt"
POS_FULL_UNIQUE = pos_preprocessed_path / "train_pos_full_unique.txt"

POS_SPACES = pos_preprocessed_path / 'train_pos_spaces.txt'
POS_HASHTAGS = pos_preprocessed_path / 'train_pos_hashtags.txt'
POS_CONTRACT = pos_preprocessed_path / 'train_pos_contract.txt'
POS_SMILEYS = pos_preprocessed_path / 'train_pos_smileys.txt'
POS_HOOKS = pos_preprocessed_path / 'train_pos_hooks.txt'

NEG_UNIQUE = neg_preprocessed_path / "train_neg_unique.txt"
NEG_FULL = "../data/train_neg_full.txt"
NEG_FULL_UNIQUE = neg_preprocessed_path / "train_neg_full_unique.txt"

NEG_SPACES = neg_preprocessed_path / 'train_neg_spaces.txt'
NEG_HASHTAGS = neg_preprocessed_path / 'train_neg_hashtags.txt'
NEG_CONTRACT = neg_preprocessed_path / 'train_neg_contract.txt'
NEG_SMILEYS = neg_preprocessed_path / 'train_neg_smileys.txt'
NEG_HOOKS = neg_preprocessed_path / 'train_neg_hooks.txt'

POS_PREPROCESSED = pos_preprocessed_path/ 'train_pos_preprocessed.txt'
NEG_PREPROCESSED = neg_preprocessed_path/ 'train_neg_preprocessed.txt'

# OTHER VALUES
POS_LABEL = '+1'
NEG_LABEL = '-1'

def preprocess():
    preprocess_pos()
    preprocess_neg()


def preprocess_pos():
    print('Pre-processing positive tweets...')

    in_filename = remove_duplicate_tweets(params.POS, POS_UNIQUE)
    in_filename = spaces(in_filename, POS_SPACES)
    in_filename = hashtags(in_filename, POS_HASHTAGS)
    in_filename = contractions(in_filename, POS_CONTRACT)
    in_filename = smileys(in_filename, POS_SMILEYS)
    in_filename = remove_hooks(in_filename, POS_PREPROCESSED)
    in_filename = add_label(in_filename, params.POS_LABELS, POS_LABEL)


def preprocess_neg():
    print('Pre-processing negative tweets...')
    in_filename = remove_duplicate_tweets(params.NEG, NEG_UNIQUE)
    in_filename = spaces(in_filename, NEG_SPACES)
    in_filename = hashtags(in_filename, NEG_HASHTAGS)
    in_filename = contractions(in_filename, NEG_CONTRACT)
    in_filename = smileys(in_filename, NEG_SMILEYS)
    in_filename = remove_hooks(in_filename, NEG_PREPROCESSED)
    in_filename = add_label(in_filename, params.NEG_LABELS, NEG_LABEL)


def remove_duplicate_tweets(in_filename, out_filename):
    lines_seen = set()
    outfile = open(out_filename, "w")
    for line in open(in_filename, "r"):
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    print('\tRemove duplicates ok.')
    return out_filename


def remove_both_duplicate_tweets(in_filename, out_filename, out_label_filename):
    line_to_occ = {}

    # Populate the dictionary with tweets and occurences
    for line in open(in_filename, "r"):
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


def spaces(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub(' +', ' ', tweet))
    outfile.close()
    print('\tSpaces ok.')
    return out_filename


def hashtags(in_filename, out_filename):
    load()
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
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
    print('\tHashtags ok.')
    return out_filename


def autocorrect(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(' '.join([spell(w) for w in tweet.split()]))
    outfile.close()
    print('\tAuto-correct ok.')
    return out_filename


def contractions(in_filename, out_filename):
    outfile = open(out_filename, "w")
    contractions = dictionaries.load_dict_contractions()
    for tweet in open(in_filename, "r"):
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
    print('\tContractions ok.')
    return out_filename


def smileys(in_filename, out_filename):
    outfile = open(out_filename, "w")
    smileys = dictionaries.load_dict_smileys()
    for tweet in open(in_filename, "r"):
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
    print('\tSmileys ok.')
    return out_filename


def numbers(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub('[-+]?\d*\.\d+|\d+', '<number>', tweet))
    outfile.close()
    print('\tNumbers ok.')
    return out_filename


def remove_hooks(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub(' *<.*?> *', '', tweet))
    outfile.close()
    print('\tHooks ok.')
    return out_filename


def punctuation(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        tweet_blob = TextBlob(tweet)
        outfile.write(' '.join(tweet_blob.words))
        outfile.write('\n')
    outfile.close()
    print('\tPunctuation ok.')
    return out_filename


# TODO: CHANGE THIS, stopwords contain negative words
def stopw(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        tweet_list = tweet.split()
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords]
        outfile.write(' '.join(clean_mess))
        outfile.write('\n')
    outfile.close()
    print('\tStopwords ok.')
    return out_filename


def normalization(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        tweet_list = tweet.split()
        for word in tweet_list:
            normalized_text = lem.lemmatize(word, 'v')
            normalized_tweet.append(normalized_text)
        outfile.write(' '.join(normalized_tweet))
        outfile.write('\n')
    outfile.close()
    print('\tNormalization ok.')
    return out_filename


def add_label(in_filename, out_filename, label_value):
    outfile = open(out_filename, 'w')
    for line in open(in_filename, 'r'):
        outfile.write(label_value)
        outfile.write(line)
    outfile.close()
    return out_filename


if __name__ == '__main__':
    preprocess()
