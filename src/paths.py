import os
import sys
from pathlib import Path
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
import src.params as params

# Global path parameters
PATH = Path(BASE_PATH)
DATA_PATH = PATH / 'data'
STANFORD_PATH = DATA_PATH / 'glove.twitter.27B'
PREPROCESSED_PATH = DATA_PATH / 'preprocessed'
POS_PREPROCESSED_PATH = PREPROCESSED_PATH / 'pos'
NEG_PREPROCESSED_PATH = PREPROCESSED_PATH / 'neg'
TEST_PREPROCESSED_PATH = PREPROCESSED_PATH / 'test'

# Before pre-processing
if params.FULL:
    POS = DATA_PATH / 'train_pos_full.txt'
    NEG = DATA_PATH / 'train_neg_full.txt'

else:
    POS = DATA_PATH / 'train_pos.txt'
    NEG = DATA_PATH / 'train_neg.txt'

# Pre-processing
if params.FULL:
    POS_UNIQUE = POS_PREPROCESSED_PATH / "train_pos_full_unique.txt"
else:
    POS_UNIQUE = POS_PREPROCESSED_PATH / "train_pos_unique.txt"

POS_SPACES = POS_PREPROCESSED_PATH / 'train_pos_spaces.txt'
POS_HASHTAGS = POS_PREPROCESSED_PATH / 'train_pos_hashtags.txt'
POS_CONTRACT = POS_PREPROCESSED_PATH / 'train_pos_contract.txt'
POS_SMILEYS = POS_PREPROCESSED_PATH / 'train_pos_smileys.txt'
POS_HOOKS = POS_PREPROCESSED_PATH / 'train_pos_hooks.txt'

if params.FULL:
    NEG_UNIQUE = NEG_PREPROCESSED_PATH / "train_neg_full_unique.txt"
else:
    NEG_UNIQUE = NEG_PREPROCESSED_PATH / "train_neg_unique.txt"

NEG_SPACES = NEG_PREPROCESSED_PATH / 'train_neg_spaces.txt'
NEG_HASHTAGS = NEG_PREPROCESSED_PATH / 'train_neg_hashtags.txt'
NEG_CONTRACT = NEG_PREPROCESSED_PATH / 'train_neg_contract.txt'
NEG_SMILEYS = NEG_PREPROCESSED_PATH / 'train_neg_smileys.txt'
NEG_HOOKS = NEG_PREPROCESSED_PATH / 'train_neg_hooks.txt'

# After pre-processing
POS_PREPROCESSED = POS_PREPROCESSED_PATH / 'train_pos_preprocessed.txt'
NEG_PREPROCESSED = NEG_PREPROCESSED_PATH / 'train_neg_preprocessed.txt'

TRAIN = PREPROCESSED_PATH / 'train.txt'
TRAIN_UNIQUE = PREPROCESSED_PATH / 'train_unique.txt'
TRAIN_CONCAT_LABEL = PREPROCESSED_PATH / "train_label.txt"
TRAIN_CONCAT_LABEL_UNIQUE = PREPROCESSED_PATH / "train_label_unique.txt"

POS_LABELS = POS_PREPROCESSED_PATH / 'train_pos_labels.txt'
NEG_LABELS = NEG_PREPROCESSED_PATH / 'train_neg_labels.txt'

# Vocabulary
VOCAB = PREPROCESSED_PATH / 'vocab.txt'
VOCAB_PICKLE = PREPROCESSED_PATH / 'vocab.pkl'
CUT_VOCAB = PREPROCESSED_PATH / "cut_vocab.txt"

# Co-occurence matrix
COOC_PICKLE = PREPROCESSED_PATH / 'cooc.pkl'

# GLOVE
EMBEDDINGS = PREPROCESSED_PATH / 'embeddings.npy'

# STANFORD EMBEDDINGS
STANFORD_NAME = 'glove.twitter.27B.' + str(params.STANFORD_K) + 'd'
STANFORD_EMBEDDINGS_TXT = STANFORD_PATH / (STANFORD_NAME + '.txt')
STANFORD_EMBEDDINGS_PICKLE = STANFORD_PATH / (STANFORD_NAME + '.pkl')
STANFORD_EMBEDDINGS_CUT_VOCAB = STANFORD_PATH / (STANFORD_NAME + '_cut_vocab_.npy')
STANFORD_NEW_CUT_VOCAB = PREPROCESSED_PATH / "stanford_cut_vocab.txt"

# TWEET EMBEDDINGS
TWEET_EMBEDDINGS = PREPROCESSED_PATH / 'tweet_embeddings.npy'

# Prediction
TEST = DATA_PATH / 'test_data.txt'
TEST_WITHOUT_INDICES = TEST_PREPROCESSED_PATH / 'test_without_indices.txt'
TEST_SPACES = TEST_PREPROCESSED_PATH / 'test_pos_spaces.txt'
TEST_HASHTAGS = TEST_PREPROCESSED_PATH / 'test_pos_hashtags.txt'
TEST_CONTRACT = TEST_PREPROCESSED_PATH / 'test_pos_contract.txt'
TEST_SMILEYS = TEST_PREPROCESSED_PATH / 'test_pos_smileys.txt'
TEST_HOOKS = TEST_PREPROCESSED_PATH / 'test_pos_hooks.txt'
TEST_PREPROCESSED = TEST_PREPROCESSED_PATH / 'test_preprocessed.txt'
TEST_EMBEDDINGS = TEST_PREPROCESSED_PATH / 'embeddings.npy'
LABEL_PREDICTIONS = TEST_PREPROCESSED_PATH / 'predictions.csv'
BEST_MODEL = PREPROCESSED_PATH / 'best_model.hd5'
