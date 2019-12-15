import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH+'/..')

from tensorflow import keras
import keras.backend as K_backend
import tensorflow as tf
import numpy as np

import src.paths as paths
import src.params as params
import src.prediction.predict as predict

import matplotlib
import matplotlib.pyplot as plt

K = params.STANFORD_K
MAX_SEQUENCE_LENGTH = 100
NO_CLASSES = 2


# Refer to: https://github.com/richliao/textClassifier/issues/28
class AttLayer(keras.layers.Layer):
    def __init__(self, attention_dim):
        self.init = keras.initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K_backend.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K_backend.variable(self.init((self.attention_dim,)))
        self.u = K_backend.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K_backend.tanh(K_backend.bias_add(K_backend.dot(x, self.W), self.b))
        ait = K_backend.dot(uit, self.u)
        ait = K_backend.squeeze(ait, -1)

        ait = K_backend.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K_backend.cast(mask, K_backend.floatx())
        ait /= K_backend.cast(K_backend.sum(ait, axis=1, keepdims=True) + K_backend.epsilon(), K_backend.floatx())
        ait = K_backend.expand_dims(ait)
        weighted_input = x * ait
        output = K_backend.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


labels = np.asarray([int(l[:-1]) for l in open(paths.TRAIN_CONCAT_LABEL_UNIQUE, 'r')])
tokenized_tweets = [t[:-1] for t in open(paths.TRAIN_UNIQUE, 'r')]

new_cut_vocab = [w[:-1] for w in open(paths.STANFORD_NEW_CUT_VOCAB, 'r')]
vocab_to_index = dict(zip(new_cut_vocab, range(len(new_cut_vocab))))

all_tweet_seqs = []
for t in tokenized_tweets:
    tweet_seq = []
    for w in t.split():
        word_i = vocab_to_index.get(w, None)
        if word_i is not None:
            tweet_seq.append(word_i)
    all_tweet_seqs.append(tweet_seq)

data = keras.preprocessing.sequence.pad_sequences(all_tweet_seqs, maxlen=MAX_SEQUENCE_LENGTH)

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
print('Total %s word vectors in cut_vocab.' % len(embeddings_cut_vocab))

embedding_matrix = np.random.random((len(embeddings_cut_vocab) + 1, K))
for word, i in vocab_to_index.items():
    word_i = vocab_to_index.get(word, None)
    if word_i is not None:
        embedding_matrix[i] = embeddings_cut_vocab[word_i]

embedding_layer = keras.layers.Embedding(len(embeddings_cut_vocab) + 1,
                            K,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

sequence_input = keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

#
# # Bidirectional LSTM
# def biLSTM():
#     embedded_sequences = embedding_layer(sequence_input)
#     l_lstm = Bidirectional(LSTM(128))(embedded_sequences)
#     preds = Dense(NO_CLASSES, activation='softmax')(l_lstm)
#     model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['acc'])
#
#     print("model fitting - Bidirectional LSTM")
#     model.summary()
#
#     model.fit(x_train, y_train,
#               nb_epoch=15, batch_size=64)
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1] * 100))
#
#     output_test = model.predict(x_test)
#     final_pred = np.argmax(output_test, axis=1)
#     org_y_label = [np.where(r == 1)[0][0] for r in y_test]
#     print(org_y_label)
#     results = confusion_matrix(org_y_label, final_pred)
#     print(results)
#     precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
#     print("Classify Glove Bi-LSTM Precision =", precisions)
#     print("Classify Glove Bi-LSTM Recall=", recall)
#     print("Classify Glove Bi-LSTM F1 Score =", f1_score)
#
#     pred_indices = np.argmax(output_test, axis=1)
#     classes = np.array(range(0, 8))
#     preds = classes[pred_indices]
#     print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
#     print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))
#
#
# def biGRUAttlayer():
#     embedded_sequences = embedding_layer(sequence_input)
#     l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
#     l_att = AttLayer(64)(l_gru)
#     preds = Dense(NO_CLASSES, activation='softmax')(l_att)
#     model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['acc'])
#
#     print("model fitting - attention GRU network")
#     model.summary()
#     model.fit(x_train, y_train,
#               epochs=15, batch_size=64)
#
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1]*100))
#
#     output_test = model.predict(x_test)
#     final_pred = np.argmax(output_test, axis=1)
#     org_y_label = [np.where(r == 1)[0][0] for r in y_test]
#     print(org_y_label)
#     results = confusion_matrix(org_y_label, final_pred)
#     print(results)
#     precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
#     print("Classify Glove Bi-LSTM Attention Precision =", precisions)
#     print("Classify Glove Bi-LSTM Attention Recall=", recall)
#     print("Classify Glove Bi-LSTM Attention F1 Score =", f1_score)
#
#     pred_indices = np.argmax(output_test, axis=1)
#     classes = np.array(range(0, NO_CLASSES))
#     preds = classes[pred_indices]
#     print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
#     print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))
#
#
# def biLSTMAttlayer():
#     embedded_sequences = embedding_layer(sequence_input)
#     l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
#     l_att = AttLayer(64)(l_gru)
#     preds = Dense(NO_CLASSES, activation='softmax')(l_att)
#     model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['acc'])
#
#     print("model fitting - attention GRU network")
#     model.summary()
#     model.fit(x_train, y_train,
#               nb_epoch=15, batch_size=64)
#
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1]*100))
#
#     output_test = model.predict(x_test)
#     final_pred = np.argmax(output_test, axis=1)
#     org_y_label = [np.where(r == 1)[0][0] for r in y_test]
#     print(org_y_label)
#     results = confusion_matrix(org_y_label, final_pred)
#     print(results)
#     precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
#     print("Classify Glove Bi-LSTM Attention Precision =", precisions)
#     print("Classify Glove Bi-LSTM Attention Recall=", recall)
#     print("Classify Glove Bi-LSTM Attention F1 Score =", f1_score)
#
#     pred_indices = np.argmax(output_test, axis=1)
#     classes = np.array(range(0, 8))
#     preds = classes[pred_indices]
#     print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
#     print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))
#
#
# def biLSTMAttDlayer():
#     embedded_sequences = embedding_layer(sequence_input)
#     l_gru = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
#     l_att = AttLayer(64)(l_gru)
#     l_att = Dense(256, activation="relu")(l_att)
#     l_att = Dropout(0.25)(l_att)
#     preds = Dense(NO_CLASSES, activation='softmax')(l_att)
#     model = Model(sequence_input, preds)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['acc'])
#
#     print("model fitting - attention GRU network")
#     model.summary()
#     model.fit(x_train, y_train,
#               nb_epoch=15, batch_size=64)
#
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1]*100))
#
#     output_test = model.predict(x_test)
#     final_pred = np.argmax(output_test, axis=1)
#     org_y_label = [np.where(r == 1)[0][0] for r in y_test]
#     print(org_y_label)
#     results = confusion_matrix(org_y_label, final_pred)
#     print(results)
#     precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
#     print("Classify Glove Bi-GRU Attention Precision =", precisions)
#     print("Classify Glove Bi-GRU Attention Recall=", recall)
#     print("Classify Glove Bi-GRU Attention F1 Score =", f1_score)
#
#     pred_indices = np.argmax(output_test, axis=1)
#     classes = np.array(range(0, NO_CLASSES))
#     preds = classes[pred_indices]
#     print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
#     print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))
#


def CNN():

    model = keras.Sequential()
    # model.add(layers.Embedding(len(vocab_to_index) + 1, K, weights=[embedding_matrix],
    #                            input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    model.add(embedding_layer)
    model.add(keras.layers.Conv1D(128, 5, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(NO_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train,
                        nb_epoch=15, batch_size=64,
                        validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)

    output_test = model.predict(x_test)
    final_pred = np.argmax(output_test, axis=1)
    org_y_label = [np.where(r == 1)[0][0] for r in y_test]
    print(org_y_label)
    results = keras.confusion_matrix(org_y_label, final_pred)
    print(results)
    precisions, recall, f1_score, true_sum = keras.metrics.precision_recall_fscore_support(org_y_label, final_pred)
    print("Classify Glove Bi-LSTM Precision =", precisions)
    print("Classify Glove Bi-LSTM Recall=", recall)
    print("Classify Glove Bi-LSTM F1 Score =", f1_score)

    pred_indices = np.argmax(output_test, axis=1)
    classes = np.array(range(0, 8))
    preds = classes[pred_indices]
    print('Log loss: {}'.format(keras.log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
    print('Accuracy: {}'.format(keras.accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


# def autoEncodeDecodeLayer():
#     input_dim = x_train.shape[1]
#
#     input_layer = Input(shape=(input_dim,))
#     encoder = Dense(32, activation="relu",
#                     activity_regularizer=regularizers.l1(10e-5))(input_layer)
#     decoder = Dense(64, activation="relu",  activity_regularizer=regularizers.l1(10e-5))(encoder)
#     decoder = Dense(128, activation='relu',  activity_regularizer=regularizers.l1(10e-5))(decoder)
#     decoder = Dense(256, activation='relu',  activity_regularizer=regularizers.l1(10e-5))(decoder)
#     decoder = Dense(NO_CLASSES, activation='softmax')(decoder)
#     autoencoder = Model(inputs=input_layer, outputs=decoder)
#     autoencoder.summary()
#
#     nb_epoch = 15
#     batch_size = 128
#     autoencoder.compile(optimizer='adam',
#                         loss='categorical_crossentropy',
#                         metrics=['acc'])
#
#     model = autoencoder.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size)
#
#     scores = autoencoder.evaluate(x_test, y_test, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1] * 100))
#
#     output_test = autoencoder.predict(x_test)
#     final_pred = np.argmax(output_test, axis=1)
#     org_y_label = [np.where(r == 1)[0][0] for r in y_test]
#     print(org_y_label)
#     results = confusion_matrix(org_y_label, final_pred)
#     print(results)
#     precisions, recall, f1_score, true_sum = metrics.precision_recall_fscore_support(org_y_label, final_pred)
#     print("Classify AutoEncoder Precision =", precisions)
#     print("Classify AutoEncoder Recall=", recall)
#     print("Classify AutoEncoder F1 Score =", f1_score)
#
#     pred_indices = np.argmax(output_test, axis=1)
#     classes = np.array(range(0, 8))
#     preds = classes[pred_indices]
#     print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], output_test)))
#     print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))


if __name__ == '__main__':
    # autoEncodeDecodeLayer()
    # biLSTMAttDlayer()
    # biLSTM()
    # biGRUAttlayer()
    # biLSTMAttlayer()
    CNN()
