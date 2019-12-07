# Rather we want to run our algorithm on the whole dataset or not
FULL = False


CUT_VOCAB_N = 5

# ----------------------------------------------------------------------------------------------------------------------
#                                                WORD EMBEDDINGS

# GLOVE
GLOVE_K = 150
GLOVE_NMAX = 100
GLOVE_ETA = 0.001
GLOVE_ALPHA = 3 / 4
GLOVE_N_EPOCHS = 10

# ALS
ALS_MAX_ITER = 10
ALS_RANK = 10
ALS_REG = 1
ALS_LAMBDA_USER = 0.1
ALS_LAMBDA_ITEM = 0.7
ALS_STOP_CRITERION = 1e-4

# STANFORD
STANFORD_K = 200  # Possible values: 25, 50, 100, 200
# ----------------------------------------------------------------------------------------------------------------------
#                                               PREDICTION

TEST_SIZE = 0.1

# LOGISTIC REGRESSION
CV_FOLDS = 5

# NEURAL-NETWORK
NN_N_EPOCHS = 20
NN_BATCH_SIZE = 100
NN_VERBOSE = True
