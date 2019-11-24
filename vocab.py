from nltk.tokenize import TweetTokenizer
import src.params as params


def concat_files(filenames):
    with open(params.TRAIN_CONCAT, 'w') as outfile:
        for filname in filenames:
            with open(filname) as infile:
                for line in infile:
                    outfile.write(line)


def build_vocab(in_filename, out_filename):
    # Retrieve tokens in the input file
    tknzr = TweetTokenizer()
    with open(in_filename, 'r') as f:
        data = f.read()
        tokens = tknzr.tokenize(data)

    # Write each token in the output file
    output_file = open(out_filename, 'w')
    for token in tokens:
        output_file.write(token+'\n')
    output_file.close()
