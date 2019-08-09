import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, classification_report


""" This function allows converting a list of string labels into one-hot encoding
    It is useful when we are working with different types of deep learning algorithms """


def encodeStringLabel(string_label):

    # e.g. string_label = ["A", "B", "C", "C", ... , "A"]

    # Instantiate the encoder
    encoder = LabelEncoder()

    # Fit label encoder
    encoder.fit(string_label)

    # Encode labels into integers
    encoded_string_label = encoder.transform(string_label)  # e.g. [0, 1, 2, 2, ..., 0]

    # Encode integers into one hot
    encoded_one_hot = np_utils.to_categorical(encoded_string_label)  # e.g. [[1, 0, 0],
                                                                     #       [0, 1, 0],
                                                                     #       [0, 0, 1],
                                                                     #       [0, 0, 1],
                                                                     #       ... ,
                                                                     #       [1, 0, 0]]

    return encoder, encoded_string_label, encoded_one_hot


""" This function allows converting a list of one-hot encoding into a list of string label
    It is useful when we are working with different types of deep learning algorithms """


def decodeOneHotLabel(encoder, encoded_one_hot):

    # e.g. encoded_one_hot = [[1, 0, 0],
    #                         [0, 1, 0],
    #                         [0, 0, 1],
    #                         [0, 0, 1],
    #                         ... ,
    #                         [1, 0, 0]]

    # Decode a list of one-hot encoding into integers
    encoded_string_label = np.argmax(encoded_one_hot, axis=1)  # e.g. [0, 1, 2, 2, ..., 0]

    # Decode the list of integers into a list of string labels
    string_label = encoder.inverse_transform(encoded_string_label)  # e.g. string_label = ["A", "B", "C", "C", ... , "A"]

    return encoded_string_label, string_label


""" Compute performance metrics of classification algorithm """


def performanceMetrics(y_true, y_predicted, target_names=None):

    accuracy = accuracy_score(y_true, y_predicted)
    balanced_accuracy = balanced_accuracy_score(y_true, y_predicted)

    if target_names is None:

        report = classification_report(y_true, y_predicted)

    elif target_names is not None:

        report = classification_report(y_true, y_predicted, target_names=target_names, digits=4)
        print(report)

    cohen_kappa = cohen_kappa_score(y_true, y_predicted)

    return accuracy, balanced_accuracy, report, cohen_kappa


""" Plot the history performances of Keras Neural Network Algorithms """


def plotKNNPerformances(history):

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy of the Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss of the Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
