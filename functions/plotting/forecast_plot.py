import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def plot_forecast(best_model: dict) -> None:
    """
    Plots the loss and validation loss by epoch for the best model.

    Parameters:
    best_model (dict): The best model and its history returned by get_best_model.

    Returns:
    None
    """

    _, history = list(best_model.items())[0]

    _, ax = plt.subplots()

    pd.Series(history.history['loss']).plot(
        style='-', color='blue',
        title='Loss by Epoch',
        ax=ax, label='loss'
    )

    pd.Series(history.history['val_loss']).plot(
        style='-', color='orange',
        ax=ax, label='val_loss'
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')

    ax.legend()

    plt.show()


def plot_actual_vs_predicted(best_model: dict, X: np.ndarray, Y: np.ndarray, scaler: MinMaxScaler or StandardScaler, column_names: list) -> None:
    """
    Plots the actual and predicted values for each feature in the dataset.

    Parameters:
    best_model (dict): The best model and its history returned by get_best_model.
    X (np.ndarray): The data (either training or validation).
    Y (np.ndarray): The labels (either training or validation).
    scaler (MinMaxScaler or StandardScaler): The scaler used to scale the data.
    column_names (list): The names of the columns (features).

    Returns:
    None
    """

    # Extract the model name and history from the best_model dictionary
    model_name, history = list(best_model.items())[0]

    # Get the predictions on the data
    Predict = history.model.predict(X)

    # Reverse the scaling
    Y = scaler.inverse_transform(Y)
    Predict = scaler.inverse_transform(Predict)

    num_columns = Y.shape[1]

    # Create subplots for each column
    fig, axes = plt.subplots(num_columns, 1, figsize=(
        15, 5*num_columns), sharex=True)

    # Loop through each column and plot the actual vs. predicted values
    for col in range(num_columns):
        actual = Y[:, col]
        predicted = Predict[:, col]

        # Plot actual values in blue
        axes[col].plot(actual, label='Actual', color='blue')

        # Plot predicted values in orange
        axes[col].plot(predicted, label='Predicted', color='orange')

        # Add labels and legends
        axes[col].set_title(column_names[col])  # Set title to column name
        axes[col].set_xlabel('Sample')
        axes[col].set_ylabel('Value')
        axes[col].legend()

    # Adjust layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()
