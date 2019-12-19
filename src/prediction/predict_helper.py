import numpy as np
import csv
import matplotlib.pyplot as plt


def plot_history(history):
    """
    Plot the evolution of the train/test accuracy/loss.

    :param history: the variable storing the evolution of accuracy/loss for both the train and the test set
    """
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


def write_predictions_in_csv(label_predictions, predictions_file_path):
    """
    Write the label predictions in the file with path predictions_file_path as a .csv.

    :param label_predictions: the labels to be written
    """
    labels_with_ids = ['Id,Prediction']
    for i in np.arange(len(label_predictions)):
        labels_with_ids.append(str(i + 1) + ',' + str(label_predictions[i]))

    with open(predictions_file_path, 'w+') as result_file:
        wr = csv.writer(result_file, delimiter=',')
        wr.writerows([x.split(',') for x in labels_with_ids])


def labels_list(labels_file_path):
    """
    Take a text file containing labels and convert it into a list of labels.
    :param labels_file_path: the path to the file containing the labels
    :return: the corresponding list of labels
    """
    with open(labels_file_path, 'r') as train_labels_file:
        return [int(label[:-1]) for label in train_labels_file]


def transform_labels(y):
    """
    Transform a list of labels with values in {-1, 1} to a list of corresponding labels in {0, 1}.

    :param y: the list of labels to transform
    :return: the transformed list of labels, i.e a list of values in {0, 1}
    """
    return [(x + 1) / 2 for x in y]


def inv_transform_labels(y_transformed):
    """
    Transform back a list of labels with values in {0, 1} to a list of corresponding labels in {-1, 1}.
    :param y_transformed: the list of labels to transform back
    :return: the inversely transformed list of labels, i.e a list of values in {-1, 1}
    """
    return [int(2*x-1) for x in y_transformed]
