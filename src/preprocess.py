import re
import os
import sys
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordsegment import load, segment
from autocorrect import Speller

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.dictionaries as dictionaries
import src.paths as paths

stopwords = set(stopwords.words('english'))
spell = Speller(lang='en')

# OTHER VALUES
POS_LABEL = '+1'
NEG_LABEL = '-1'


def preprocess():
    preprocess_pos()
    preprocess_neg()
    # Concatenate the proprocessed versions of positive tweets and negative tweets into a new file
    concat_files([paths.POS_LABELS, paths.NEG_LABELS], paths.TRAIN)
    # Remove the tweets that appear >= 2 times and separate label from tweet.
    remove_both_duplicate_tweets(paths.TRAIN, paths.TRAIN_UNIQUE, paths.TRAIN_CONCAT_LABEL_UNIQUE)


def preprocess_pos():
    print('Pre-processing positive tweets...')

    in_filename = remove_duplicate_tweets(paths.POS, paths.POS_UNIQUE)
    in_filename = spaces(in_filename, paths.POS_SPACES)
    in_filename = hashtags(in_filename, paths.POS_HASHTAGS)
    in_filename = contractions(in_filename, paths.POS_CONTRACT)
    in_filename = smileys(in_filename, paths.POS_SMILEYS)
    in_filename = remove_hooks(in_filename, paths.POS_PREPROCESSED)
    in_filename = add_label(in_filename, paths.POS_LABELS, POS_LABEL)


def preprocess_neg():
    print('Pre-processing negative tweets...')
    in_filename = remove_duplicate_tweets(paths.NEG, paths.NEG_UNIQUE)
    in_filename = spaces(in_filename, paths.NEG_SPACES)
    in_filename = hashtags(in_filename, paths.NEG_HASHTAGS)
    in_filename = contractions(in_filename, paths.NEG_CONTRACT)
    in_filename = smileys(in_filename, paths.NEG_SMILEYS)
    in_filename = remove_hooks(in_filename, paths.NEG_PREPROCESSED)
    in_filename = add_label(in_filename, paths.NEG_LABELS, NEG_LABEL)


def remove_duplicate_tweets(in_filename, out_filename):
    print('\tRemoving duplicate tweets...')
    lines_seen = set()
    outfile = open(out_filename, "w")
    for line in open(in_filename, "r"):
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()
    print('\t\tRemove duplicates ok.')
    return out_filename


def remove_both_duplicate_tweets(in_filename, out_filename, out_label_filename):
    print('Removing both duplicates...')
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
    print('\tHandling spaces...')
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub(' +', ' ', tweet))
    outfile.close()
    print('\t\tSpaces ok.')
    return out_filename


def hashtags(in_filename, out_filename):
    print('\tHandling hashtags...')
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
    print('\t\tHashtags ok.')
    return out_filename


def autocorrect(in_filename, out_filename):
    print('\tAuto-correcting tweets...')
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(' '.join([spell(w) for w in tweet.split()]))
    outfile.close()
    print('\t\tAuto-correct ok.')
    return out_filename


def contractions(in_filename, out_filename):
    print('\tHandling contractions...')
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
    print('\t\tContractions ok.')
    return out_filename


def smileys(in_filename, out_filename):
    print('\tHandling smileys...')
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
    print('\t\tSmileys ok.')
    return out_filename


def numbers(in_filename, out_filename):
    print('\tHandling numbers...')
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub('[-+]?\d*\.\d+|\d+', '<number>', tweet))
    outfile.close()
    print('\t\tNumbers ok.')
    return out_filename


def remove_hooks(in_filename, out_filename):
    print('\tRemoving hooks...')
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub(' *<.*?> *', '', tweet))
    outfile.close()
    print('\t\tHooks ok.')
    return out_filename


def punctuation(in_filename, out_filename):
    print('\tHandling punctuation...')
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        tweet_blob = TextBlob(tweet)
        outfile.write(' '.join(tweet_blob.words))
        outfile.write('\n')
    outfile.close()
    print('\t\tPunctuation ok.')
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
    print('\t\tStopwords ok.')
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
    print('\t\tNormalization ok.')
    return out_filename


def add_label(in_filename, out_filename, label_value):
    outfile = open(out_filename, 'w')
    for line in open(in_filename, 'r'):
        outfile.write(label_value)
        outfile.write(line)
    outfile.close()
    return out_filename


def concat_files(in_filenames, out_filename):
    with open(out_filename, 'w') as outfile:
        for filename in in_filenames:
            with open(filename) as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == '__main__':
    preprocess()
