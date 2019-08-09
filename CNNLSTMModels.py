from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten

# https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826


def CNNLSTMModelV1(input_shape, classes):

    # Instantiate a Neural Network
    model = Sequential()

    # Add Convolutional layer(s)
    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=2, activation="relu"), input_shape=(None, input_shape[1], input_shape[2])))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
    model.add(TimeDistributed(Flatten()))

    # Add LSTM layer(s)
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(200))

    # Add a Dropout
    model.add(Dropout(0.5))

    model.add(Dense(250, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))

    # Add a regular densely-connected Neural Network layer
    model.add(Dense(classes, activation='softmax'))

    return model
