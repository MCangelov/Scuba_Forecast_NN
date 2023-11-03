import numpy as np
import time

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping, History, TensorBoard
from keras import Model, backend as K
from typing import Dict, Any


def generate_models(layers: list, units: list, window: int, features: int, dropout: list) -> dict:
    """
    This function generates a dictionary of LSTM models with different configurations.

    Parameters:
    layers (list): A list of integers representing the number of layers in the LSTM model.
    units (list): A list of integers representing the number of units in each layer of the LSTM model.
    window (int): The size of the window for the LSTM model.
    features (int): The number of features for the LSTM model.
    dropout (list): A list of floats representing the dropout rate for each layer in the LSTM model.

    Returns:
    dict: A dictionary where each key is a string representing the model configuration and each value is another dictionary with 'model' and 'history' keys. 'model' is an instance of the LSTM model and 'history' is None.
    """

    models = {}

    for n_layers in layers:
        for n_units in units:
            for drop in dropout:
                K.clear_session()

                model_name = f'{n_layers} layers, {n_units} units, dropout {drop}'
                model = create_multiple_LSTM(
                    n_layers=n_layers, units=n_units, window=window, features=features, dropout=drop)
                models[model_name] = {
                    'model': model,
                    'history': None
                }

                print(f"Summary of the model: {model_name}")
                model.summary()
                print("\n")

    return models


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

    model.add(Dense(361, activation='relu'))
    model.add(Dense(features))

    model.compile(loss='root_mean_squared_error', optimizer='adam', metrics=[
                  'mean_absolute_error', 'root_mean_squared_error'])

    return model


def train_model(model: Model,
                trainX: np.ndarray, trainY: np.ndarray,
                valX: np.ndarray, valY: np.ndarray,
                epochs: int = 500,
                patience: int = 5,
                batch_size: int = 16,
                verbose: int = 1,
                use_tensorboard: bool = False) -> History:
    """
    Trains the provided model using the given training and validation data.

    Parameters:
    model (Model): The Keras model to train.
    trainX (np.ndarray): The training data.
    trainY (np.ndarray): The training labels.
    valX (np.ndarray): The validation data.
    valY (np.ndarray): The validation labels.
    epochs (int, Optional): The number of epochs to train for. Defaults to 500.
    patience (int, Optional): The number of epochs to wait for improvement before stopping. Defaults to 5.
    batch_size (int, Optional): The batch size for training. Defaults to 16.
    verbose (int, Optional): The level of verbosity. Defaults to 1.
    use_tensorboard (bool, Optional): Whether to use TensorBoard callback. Defaults to False.

    Returns:
    History: The training history.
    """

    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
                                   verbose=1, mode='auto', restore_best_weights=True)

    callbacks = [early_stopping]

    if use_tensorboard:
        # TensorBoard callback
        log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard_callback)  # type: ignore

    history = model.fit(trainX, trainY, validation_data=(valX, valY),
                        shuffle=False, epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose, callbacks=callbacks)  # type: ignore

    return history


def get_best_model(models: Dict[str, Dict[str, Any]], metric: str = 'root_mean_squared_error') -> Dict[str, Dict[str, Any]]:
    """
    This function calculates the metrics for all models, prints them in a table format, and finds the best model based on a specified metric.

    Args:
        models (Dict[str, Dict[str, Any]]): A dictionary containing model names as keys and their information as values.
        metric (str, optional): The metric to be used for comparison. Defaults to 'root_mean_squared_error'.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary with the best model and its history.
    """
    best_model_name = ''
    best_model_history = None
    best_model = None
    best_metric_value = float('inf')

    # Print the column names
    print(f"{'Model':<20} {'Dropout':<10} {'RMSE':<10} {'MAE':<10} {'Val RMSE':<10} {'Val MAE':<10}")

    for full_model_name, model_info in models.items():
        if model_info['history'] is not None:
            # Get the metrics for the current model
            rmse = round(
                model_info['history'].history['root_mean_squared_error'][-1], 5)
            mae = round(
                model_info['history'].history['mean_absolute_error'][-1], 5)
            val_rmse = round(
                model_info['history'].history['val_root_mean_squared_error'][-1], 5)
            val_mae = round(
                model_info['history'].history['val_mean_absolute_error'][-1], 5)

            # Extract dropout from the full model name and create a new model name without dropout
            dropout_str = full_model_name.split(', dropout ')[-1]
            dropout = float(dropout_str)
            model_name = full_model_name.replace(
                ', dropout ' + dropout_str, '')

            # Find the best value for the specified metric
            metric_values = model_info['history'].history[metric]
            min_metric_value = min(metric_values)
            if min_metric_value < best_metric_value:
                best_metric_value = min_metric_value
                best_model_name = model_name  # Keep the full model name for comparison
                best_model_history = model_info['history']
                best_model = model_info['model']  # Update the best model

            # Print the model name and its metrics
            print(
                f"{model_name:<20} {dropout:<10} {rmse:<10} {mae:<10} {val_rmse:<10} {val_mae:<10}")

    return {best_model_name: {'model': best_model, 'history': best_model_history}}
