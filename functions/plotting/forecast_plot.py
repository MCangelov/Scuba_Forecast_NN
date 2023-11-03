import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def plot_forecast(best_model: dict) -> None:
    """
    Plots the loss and validation loss by epoch for the best model.

    Parameters:
    best_model (dict): The best model and its history returned by get_best_model.

    Returns:
    None
    """

    history = list(best_model.values())[0]['history']

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


def plot_metrics(best_model: dict, metric: str) -> None:
    """
    Plots the metrics and validation metrics by epoch for the best model.

    Parameters:
    best_model (dict): The best model and its history returned by get_best_model.
    metric (str): The name of the metric to plot.

    Returns:
    None
    """

    history = list(best_model.values())[0]['history']

    _, ax = plt.subplots()

    pd.Series(history.history[metric]).plot(
        style='-', color='blue',
        title=f'{metric} by Epoch',
        ax=ax, label=metric
    )

    pd.Series(history.history[f'val_{metric}']).plot(
        style='-', color='orange',
        ax=ax, label=f'val_{metric}'
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())

    ax.legend()

    plt.show()


def plot_predictions(original_data, predicted_data, column_names):
    """
    Plots the validation and predicted values for each feature.

    Parameters:
    valid_scaled (ndarray): The scaled validation data.
    persistance_valid_predictions (ndarray): The predicted values.
    column_names (List[str]): The names of the features.
    """

    for i in range(original_data.shape[1]):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=original_data[:, i],
            mode='lines',
            name='Validation',
            line=dict(color='orange')
        ))

        fig.add_trace(go.Scatter(
            y=predicted_data[:, i],
            mode='lines',
            name='Predicted',
            line=dict(color='magenta', dash='dash')
        ))

        fig.update_layout(
            title=f'{column_names[i]}',
            xaxis_title='Time Steps',
            yaxis_title='Value'
        )

        fig.show()
