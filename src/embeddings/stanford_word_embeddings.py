import sys
import os
import numpy as np
import pickle

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')
import src.paths as paths


def stanford_to_pkl(stanford_txt_filename, stanford_pkl_filename):
    """
    Store the Stanford GLOVE vectors in a pickle file.

    :param stanford_txt_filename: the path to the Stanford GLOVE text file, storing the embedding vectors trained by
    Stanford collaborators on a large Twitter dataset
    :param stanford_pkl_filename: the path to the future Stanford GLOVE pickle file
    """
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


def stanford_only_cut_vocab(stanford_pkl_path, stanford_cut_vocab_npy_path, cut_vocab_path,
                            new_cut_vocab_path):
    """
    Do the intersection between our restricted vocabulary and the Stanford vocabulary, and save it in a
    new_cut_vocab npy file.

    :param stanford_pkl_path: the path to the pickle file storing the Stanford GLOVE vectors
    :param stanford_cut_vocab_npy_path: the path to the embedding vectors stored as an npy file of the intersection
    between our current restricted vocabulary and the Stanford vocabulary
    between our current vocabulary and the words coming from the Stanford GLOVE vectors
    :param cut_vocab_path: the path to our current restricted vocabulary
    :param new_cut_vocab_path: the path to the future new vocabulary, being the intersection between our current
    restricted vocabulary and the Stanford vocabulary
    """
    print('Retrieving word embeddings from Stanford...')
    stanford_cut_vocab_list = []
    cut_vocab_file = open(cut_vocab_path, 'r')
    new_cut_vocab_file = open(new_cut_vocab_path, 'w')
    with open(stanford_pkl_path, 'rb') as f:
        stanford_embeddings = pickle.load(f)
        stanford_bin = set()
        i = 1
        for word in cut_vocab_file:
            if word[:-1] in stanford_embeddings:
                if word[:-1] not in stanford_bin:
                    stanford_cut_vocab_list.append(stanford_embeddings[word[:-1]])
                    new_cut_vocab_file.write(word)
                    stanford_bin.add(word[:-1])
            else:
                print('\t{}: {} is not part of stanford word embeddings.'.format(i, word[:-1]))
                i += 1
    cut_vocab_file.close()
    new_cut_vocab_file.close()
    np.save(stanford_cut_vocab_npy_path, np.asarray(stanford_cut_vocab_list))
    print('Stanford word embeddings ok.')


if __name__ == '__main__':
    # stanford_to_pkl(paths.STANFORD_EMBEDDINGS_TXT, paths.STANFORD_EMBEDDINGS_PICKLE)
    stanford_only_cut_vocab(paths.STANFORD_EMBEDDINGS_PICKLE, paths.STANFORD_EMBEDDINGS_CUT_VOCAB, paths.CUT_VOCAB,
                            paths.STANFORD_NEW_CUT_VOCAB)
