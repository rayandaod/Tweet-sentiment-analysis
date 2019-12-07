import sys
import os
import numpy as np
import pickle
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.paths as paths


def stanford_to_pkl(stanford_txt_filename, stanford_pkl_filename):
    print('Parsing Stanford Glove file...')
    input_file = open(stanford_txt_filename, 'r')
    word_vector_dict = dict()
    for line in input_file:
        word_and_vector = line.split()
        word = word_and_vector[0]
        vector = np.array(word_and_vector[1:]).astype(np.float)
        word_vector_dict[word] = vector
    input_file.close()

    with open(stanford_pkl_filename, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(word_vector_dict, f, pickle.HIGHEST_PROTOCOL)
    print('\tParsing ok.')


def stanford_only_cut_vocab(stanford_pkl_filename, stanford_cut_vocab_npy_filename, cut_vocab_filename,
                            new_cut_vocab_filename):
    print('Retrieving word embeddings from Stanford...')
    stanford_cut_vocab_list = []
    cut_vocab_file = open(cut_vocab_filename, 'r')
    new_cut_vocab_file = open(new_cut_vocab_filename, 'w')
    with open(stanford_pkl_filename, 'rb') as f:
        stanford_embeddings = pickle.load(f)
        i = 1
        for word in cut_vocab_file:
            if word[:-1] in stanford_embeddings:
                stanford_cut_vocab_list.append(stanford_embeddings[word[:-1]])
                new_cut_vocab_file.write(word)
            else:
                print('{}: {} is not part of stanford word embeddings.'.format(i, word[:-1]))
                i += 1
    cut_vocab_file.close()
    new_cut_vocab_file.close()
    np.save(stanford_cut_vocab_npy_filename, np.asarray(stanford_cut_vocab_list))
    print('\tStanford word embeddings ok.')


if __name__ == '__main__':
    # To be run once
    # stanford_to_pkl(paths.STANFORD_EMBEDDINGS_TXT, paths.STANFORD_EMBEDDINGS_PICKLE)

    stanford_only_cut_vocab(paths.STANFORD_EMBEDDINGS_PICKLE, paths.STANFORD_EMBEDDINGS_CUT_VOCAB, paths.CUT_VOCAB,
                            paths.STANFORD_NEW_CUT_VOCAB)
