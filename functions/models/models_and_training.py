import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping, History
from keras import Model


def create_single_LTSM(units: int, window: int, features: int, dropout: float = 0.2) -> Model:
    """
    Creates a single layer LSTM model.

    Parameters:
    units (int): The number of LSTM units.
    window (int): The length of the input sequence.
    features (int): The number of input features.
    dropout (float, optional): The dropout rate. Defaults to 0.2.

    Returns:
    Model: A compiled Keras model.
    """

    model = Sequential()
    model.add(LSTM(units=units, activation='relu',
                   input_shape=(window, features)))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(19))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[
                  'mean_absolute_error', 'root_mean_squared_error'])

    return model


def train_model(model: Model,
                trainX: np.ndarray, trainY: np.ndarray,
                valX: np.ndarray, valY: np.ndarray,
                epochs: int = 1000,
                patience: int = 400,
                batch_size: int = 16) -> History:
    """
    Trains the provided model using the given training and validation data.

    Parameters:
    model (Model): The Keras model to train.
    trainX (np.ndarray): The training data.
    trainY (np.ndarray): The training labels.
    valX (np.ndarray): The validation data.
    valY (np.ndarray): The validation labels.
    epochs (int, optional): The number of epochs to train for. Defaults to 1000.
    patience (int, optional): The number of epochs to wait for improvement before stopping. Defaults to 400.
    batch_size (int, optional): The batch size for training. Defaults to 16.

    Returns:
    History: The training history.
    """

    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=1, mode='auto', restore_best_weights=True)

    history = model.fit(trainX, trainY, validation_data=(valX, valY),
                        shuffle=False, epochs=epochs,
                        batch_size=batch_size,
                        verbose=2, callbacks=[early_stopping])

    return history
