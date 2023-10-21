import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping, History
from keras import Model


def create_multiple_LSTM(n_layers: int, units: int, window: int, features: int, dropout: float = 0.0) -> Model:
    """
    (Optionally) Creates a multi-layer LSTM model.

    Parameters:
    n_layers (int): The number of LSTM layers.
    units (int): The number of LSTM units.
    window (int): The length of the input sequence.
    features (int): The number of input features.
    dropout (float, optional): The dropout rate. Defaults to 0.0.

    Returns:
    Model: A compiled Keras model.
    """

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True if n_layers > 1 else False,
                   activation='relu', input_shape=(window, features)))
    model.add(Dropout(dropout))

    for i in range(n_layers - 1):
        model.add(LSTM(units=units, return_sequences=True if i < n_layers - 2 else False,
                       activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(features))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[
                  'mean_absolute_error', 'root_mean_squared_error'])

    return model


def train_model(model: Model,
                trainX: np.ndarray, trainY: np.ndarray,
                valX: np.ndarray, valY: np.ndarray,
                epochs: int = 500,
                patience: int = 50,
                batch_size: int = 16,
                verbose: int = 1) -> History:
    """
    Trains the provided model using the given training and validation data.

    Parameters:
    model (Model): The Keras model to train.
    trainX (np.ndarray): The training data.
    trainY (np.ndarray): The training labels.
    valX (np.ndarray): The validation data.
    valY (np.ndarray): The validation labels.
    epochs (int, Optional): The number of epochs to train for. Defaults to 500.
    patience (int, Optional): The number of epochs to wait for improvement before stopping. Defaults to 50.
    batch_size (int, Optional): The batch size for training. Defaults to 16.
    verbose (int, Optional): The level of verbosity. Defaults to 1.

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
                        verbose=verbose, callbacks=[early_stopping])  # type: ignore

    return history
