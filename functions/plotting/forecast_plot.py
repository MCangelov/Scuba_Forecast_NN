import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
import numpy as np


def plot_forecast(model: Model, testX: np.ndarray, testY: np.ndarray, history: dict) -> None:
    """
    Plots the loss by epoch and the forecast vs actual values.

    Parameters:
    model (Model): The trained Keras model.
    testX (np.ndarray): The test data.
    testY (np.ndarray): The test labels.
    history (dict): The training history.

    Returns:
    None
    """

    fig, ax = plt.subplots(2, 1)

    pd.Series(history['loss']).plot(
        style='k', alpha=0.50,
        title='Loss by Epoch',
        ax=ax[0], label='loss'
    )

    pd.Series(history['val_loss']).plot(
        style='k', ax=ax[0], label='val_loss'
    )

    ax[0].legend()

    predicted = model.predict(testX)

    pd.Series(testY.reshape(-1)).plot(
        style='k--', alpha=0.5,
        ax=ax[1],
        title='Forecast vs Actual',
        label='actual'
    )

    pd.Series(predicted.reshape(-1)).plot(
        style='k', label='Forecast', ax=ax[1]
    )

    fig.tight_layout()
    ax[1].legend()
    plt.show()
