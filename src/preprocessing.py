from textblob import TextBlob
import re


def remove_duplicate_tweets(in_filename, out_filename):
    lines_seen = set()
    outfile = open(out_filename, "w")
    for line in open(in_filename, "r"):
        if line not in lines_seen:
            outfile.write(line)
            lines_seen.add(line)
    outfile.close()


def remove_both_duplicate_tweets(in_filename, out_filename):
    line_to_occ = {}

    # Populate the dictionary with tweets and occurences
    for line in open(in_filename, "r"):
        if line in line_to_occ:
            line_to_occ[line] += 1
        else:
            line_to_occ[line] = 1

    # Remove lines that appear >= 2 times (which means they were both in the positive, and negative dataset)
    for x in list(line_to_occ):
        if line_to_occ[x] >= 2:
            line_to_occ.pop(x)

    # Write the remaining tweets in the output file
    outfile = open(out_filename, "w")
    for line in line_to_occ.keys():
        outfile.write(line)
    outfile.close()


def numbers(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub('[-+]?\d*\.\d+|\d+', '<number>', tweet))
    outfile.close()


def remove_hooks(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        outfile.write(re.sub('<[^>]+>', '', tweet))
    outfile.close()


def punctuation(in_filename, out_filename):
    outfile = open(out_filename, "w")
    for tweet in open(in_filename, "r"):
        tweet_blob = TextBlob(tweet)
        outfile.write(' '.join(tweet_blob.words))
        outfile.write('\n')
    outfile.close()
