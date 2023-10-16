from pandas import DataFrame
from typing import Tuple
import numpy as np
import pandas as pd


def create_lagged_features(dataframe: DataFrame, lag: int = 3) -> DataFrame:
    """
    Creates lagged features for a DataFrame.

    Parameters:
    dataframe (DataFrame): The input DataFrame.
    lag (int): The number of lags to create.

    Returns:
    DataFrame: A DataFrame with the lagged features.
    """
    dataframe_copy = dataframe.copy()

    for feature in dataframe.columns:
        for i in range(1, lag + 1):
            dataframe_copy[f"{feature}_lag_{i}"] = dataframe[feature].shift(i)

    dataframe_copy = dataframe_copy.dropna()

    return dataframe_copy


def split_dataframe(dataframe: DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Splits a DataFrame into training, validation, test sets and a DatetimeIndex.

    Parameters:
    dataframe (DataFrame): The input DataFrame.
    train_ratio (float): The ratio of the training set.
    valid_ratio (float): The ratio of the validation set.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]: A tuple containing the training set, validation set, test set and test index.
    """

    total_length = len(dataframe)
    train_length = int(total_length * train_ratio)
    valid_length = int(total_length * valid_ratio)

    train = dataframe.iloc[:train_length].values
    valid = dataframe.iloc[train_length:train_length + valid_length].values
    test = dataframe.iloc[train_length + valid_length:].values

    test_index = dataframe.index[train_length + valid_length:]

    return train, valid, test, test_index


def sliding_window(dataset: np.ndarray, window_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms a dataset into overlapping sequences (or "sliding windows") of a specific size.

    Parameters:
    dataset (ndarray): The input dataset as a NumPy array.
    window_size (int): The size of the sequences (or "windows").

    Returns:
    Tuple[ndarray, ndarray]: A tuple containing two NumPy arrays. The first array contains the sequences of data, and the second array contains the corresponding targets (the next step after each sequence).
    """
    x = []
    y = []

    for i in range(len(dataset) - window_size - 1):
        window = dataset[i:(i+window_size), :]
        x.append(window)
        y.append(dataset[i+window_size, :])

    x = np.array(x)
    y = np.array(y)

    return x, y
