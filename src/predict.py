import os
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

import src.params as params
import src.preprocess as prep
import src.tweet_embeddings as t_embed

preprocessed_path = Path(BASE_PATH + '/data/preprocessed')
test_preprocessed_path = preprocessed_path / 'test'

TEST_SPACES = test_preprocessed_path / 'test_pos_spaces.txt'
TEST_HASHTAGS = test_preprocessed_path / 'test_pos_hashtags.txt'
TEST_CONTRACT = test_preprocessed_path / 'test_pos_contract.txt'
TEST_SMILEYS = test_preprocessed_path / 'test_pos_smileys.txt'
TEST_HOOKS = test_preprocessed_path / 'test_pos_hooks.txt'


def predict():
    preprocess_test()
    test_embeddings = t_embed.embed(params.TEST_PREPROCESSED, params.TEST_EMBEDDINGS)
    clf = LogisticRegressionCV(cv=5, random_state=0,
                               multi_class='multinomial').fit(X, y)
    clf.predict(X[:2, :])

    clf.predict_proba(X[:2, :]).shape

    clf.score(X, y)


def train():



def preprocess_test():
    print('Pre-processing the test set...')
    in_filename = prep.spaces(params.TEST, TEST_SPACES)
    in_filename = prep.hashtags(in_filename, TEST_HASHTAGS)
    in_filename = prep.contractions(in_filename, TEST_CONTRACT)
    in_filename = prep.smileys(in_filename, TEST_SMILEYS)
    in_filename = prep.remove_hooks(in_filename, params.TEST_PREPROCESSED)


if __name__ == '__main__':
    predict()
