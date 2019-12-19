import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

from tensorflow import keras
import numpy as np

import src.paths as paths
import src.params as params
import src.prediction.predict_helper as helper
import src.prediction.AttLayer as AttLayer


# Load existing arrays
labels = np.asarray([int(l[:-1]) for l in open(paths.TRAIN_CONCAT_LABEL_UNIQUE, 'r')])
tokenized_tweets = [t[:-1] for t in open(paths.TRAIN_UNIQUE, 'r')]
new_cut_vocab = [w[:-1] for w in open(paths.STANFORD_NEW_CUT_VOCAB, 'r')]

# Buid a word-to-index dictionary (0 is reserved for padding)
word_to_index = dict(zip(new_cut_vocab, np.arange(len(new_cut_vocab)) + 1))

# Build a list of tweet sequences, i.e each tweet is represented by a list of indices of the corresponding words it
# contains
all_tweet_seqs = []
for t in tokenized_tweets:
    tweet_seq = []
    for w in t.split():
        word_i = word_to_index.get(w, None)
        if word_i is not None:
            tweet_seq.append(word_i)
    all_tweet_seqs.append(tweet_seq)

# Pad those sequences with zeros to get a matrix of shape ()
data = keras.preprocessing.sequence.pad_sequences(all_tweet_seqs, maxlen=params.MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Shuffle the training set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Split the training set
nb_validation_samples = int(params.TEST_SIZE * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

embeddings_cut_vocab = np.load(paths.STANFORD_EMBEDDINGS_CUT_VOCAB)
word_to_glove = dict(zip(new_cut_vocab, embeddings_cut_vocab))


def create_pretrained_embedding_matrix(word_to_glove, word_to_index):
    vocab_len = len(word_to_index) + 1  # adding 1 to account for masking
    embedding_dim = next(iter(word_to_glove.values())).shape[0]  # works with any glove dimensions (e.g. 50)

    emb_matrix = np.zeros((vocab_len, embedding_dim))
    for word, index in word_to_index.items():
        emb_matrix[index] = word_to_glove[word]  # create embedding: word index to Glove word embedding

    return emb_matrix


embedding_matrix = create_pretrained_embedding_matrix(word_to_glove, word_to_index)
embedding_layer = keras.layers.Embedding(len(embeddings_cut_vocab) + 1, params.STANFORD_K, weights=[embedding_matrix],
                                         input_length=params.MAX_SEQUENCE_LENGTH,
                                         trainable=params.NN_TRAIN_EMBEDDING_LAYER)


def better_predict():
    # model = build_model(lr=1e-4, lr_d=0, units=128, dr=0.5)
    model = NN_model()
    # model = CNN_model_1()
    # model = CNN_model_2()
    # model = autoEncodeDecodeLayer_model()
    # model = biLSTM_model()
    model.summary()

    # es = keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=params.NN_PATIENCE)
    # mc = keras.callbacks.ModelCheckpoint(paths.BEST_MODEL, monitor='val_acc', mode='max', verbose=1,
    #                                      save_best_only=True)

    history = model.fit(x_train, np.asarray(helper.transform_labels(y_train)), epochs=params.NN_N_EPOCHS,
                        batch_size=params.NN_BATCH_SIZE, verbose=1)

    _, accuracy = model.evaluate(x_test, np.asarray(helper.transform_labels(y_test)), verbose=params.NN_VERBOSE)

    loss, accuracy = model.evaluate(x_train, np.asarray(helper.transform_labels(y_train)), verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, np.asarray(helper.transform_labels(y_test)), verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    helper.plot_history(history)


def NN_model():
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0005,
                                                                              beta_1=0.45,
                                                                              beta_2=0.445,
                                                                              amsgrad=False), metrics=['acc'])
    return model


def biLSTM_model():
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def CNN_model_1():
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Conv1D(128, 2, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def CNN_model_2():
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Conv1D(128, 5, activation='relu'))
    model.add(keras.layers.MaxPooling1D(5))
    model.add(keras.layers.Conv1D(128, 5, activation='relu'))
    model.add(keras.layers.MaxPooling1D(5))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def autoEncodeDecodeLayer_model():
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.Dense(32, activation="relu", activity_regularizer=keras.regularizers.l1(10e-5)))
    model.add(keras.layers.Dense(64, activation="relu",  activity_regularizer=keras.regularizers.l1(10e-5)))
    model.add(keras.layers.Dense(128, activation='relu',  activity_regularizer=keras.regularizers.l1(10e-5)))
    model.add(keras.layers.Dense(256, activation='relu',  activity_regularizer=keras.regularizers.l1(10e-5)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def build_model(lr=0.0, lr_d=0.0, units=0, dr=0.0):
    input = embedding_layer
    x1 = keras.layers.SpatialDropout1D(dr)(input)

    x_gru = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units, return_sequences=True))(x1)
    x1 = keras.layers.Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = keras.layers.GlobalAveragePooling1D()(x1)
    max_pool1_gru = keras.layers.GlobalMaxPooling1D()(x1)

    x3 = keras.layers.Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = keras.layers.GlobalAveragePooling1D()(x3)
    max_pool3_gru = keras.layers.GlobalMaxPooling1D()(x3)

    x_lstm = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units, return_sequences=True))(x1)
    x1 = keras.layers.Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = keras.layers.GlobalAveragePooling1D()(x1)
    max_pool1_lstm = keras.layers.GlobalMaxPooling1D()(x1)

    x3 = keras.layers.Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = keras.layers.GlobalAveragePooling1D()(x3)
    max_pool3_lstm = keras.layers.GlobalMaxPooling1D()(x3)

    x = keras.layers.concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru, avg_pool1_lstm,
                                  max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(keras.layers.Dense(128, activation='relu')(x))
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(keras.layers.Dense(100, activation='relu')(x))
    x = keras.layers.Dense(5, activation="sigmoid")(x)
    model = keras.models.Model(inputs=input, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=lr, decay=lr_d), metrics=["acc"])
    return model


if __name__ == '__main__':
    better_predict()
