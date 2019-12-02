import os
import sys
from pathlib import Path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

# Rather we wan to run our algorithm on the whole dataset or not
FULL = True

# Global path parameters
PATH = Path(BASE_PATH)
DATA_PATH = PATH / 'data'
PREPROCESSED_PATH = DATA_PATH / 'preprocessed'

# Before pre-processing
if FULL:
    POS = DATA_PATH / 'train_pos_full.txt'
    NEG = DATA_PATH / 'train_neg_full.txt'

else:
    POS = DATA_PATH / 'train_pos.txt'
    NEG = DATA_PATH / 'train_neg.txt'

TEST = DATA_PATH / 'test_data.txt'

# After pre-processing
POS_PREPROCESSED = PREPROCESSED_PATH / 'pos/train_pos_preprocessed.txt'
NEG_PREPROCESSED = PREPROCESSED_PATH / 'neg/train_neg_preprocessed.txt'
TEST_PREPROCESSED = PREPROCESSED_PATH / 'test/test_preprocessed.txt'

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
EMBEDDINGS = PREPROCESSED_PATH / 'embeddings'


# Prediction
TEST_EMBEDDINGS = PREPROCESSED_PATH / 'test/embeddings'
