import os
import sys
from pathlib import Path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

PATH = Path(BASE_PATH)
DATA_PATH = PATH / 'data'
PREPROCESSED_PATH = DATA_PATH / 'preprocessed'

# Before pre-processing
POS = DATA_PATH / 'train_pos.txt'
NEG = DATA_PATH / 'train_neg.txt'

# After pre-processing
POS_PREPROCESSED = PREPROCESSED_PATH / 'pos/train_pos_preprocessed.txt'
NEG_PREPROCESSED = PREPROCESSED_PATH / 'neg/train_neg_preprocessed.txt'
POS_LABELS = PREPROCESSED_PATH / 'pos/train_pos_labels.txt'
NEG_LABELS = PREPROCESSED_PATH / 'neg/train_neg_labels.txt'
TRAIN_UNIQUE = PREPROCESSED_PATH / 'train_unique.txt'

# Vocabulary
VOCAB = PREPROCESSED_PATH / 'vocab.txt'
VOCAB_PICKLE = PREPROCESSED_PATH / 'vocab.pkl'
CUT_VOCAB = PREPROCESSED_PATH / "cut_vocab.txt"
CUT_VOCAB_N = 5

# Co-occurence matrix
COOC_PICKLE = PREPROCESSED_PATH / 'cooc.pkl'

# GLOVE
K = 20
NMAX = 100
ETA = 0.001
ALPHA = 3 / 4
N_EPOCHS = 10
EMBEDDINGS = PREPROCESSED_PATH / 'embeddings.npy'
TWEET_EMBEDDINGS = PREPROCESSED_PATH / 'tweet_embeddings'

