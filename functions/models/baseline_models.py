import numpy as np
from math import sqrt
from statsmodels.tsa.vector_ar.vecm import VECM
from sklearn.metrics import mean_squared_error
from functions.checks_and_preprocessing.lagging_and_splitting import sliding_window

# lag must be <= window_size


def persistence_with_lag_model(dataset, window_size, lag, column_names):
    """
    Implements a Persistence with Lag model using a sliding window approach and calculates the RMSE.

    Parameters:
    dataset (ndarray): The input dataset as a NumPy array.
    window_size (int): The size of the sequences (or "windows").
    lag (int): The number of time steps to go back for the prediction.
    column_names (List[str]): The names of the features.

    Returns:
    Tuple[ndarray, Dict[str, float], float]: A tuple containing the predictions for each sequence in the dataset, a dictionary with the RMSE for each feature, and the total RMSE.
    """

    x, y = sliding_window(dataset, window_size)

    # For each sequence, the prediction is the value at the specified lag
    predictions = x[:, -lag, :]  # type: ignore

    # Calculate the RMSE for each feature
    persistance_rmse_dict = {}
    for i in range(predictions.shape[1]):  # type: ignore
        rmse = sqrt(mean_squared_error(
            y[:, i], predictions[:, i]))  # type: ignore
        persistance_rmse_dict[column_names[i]] = rmse

    total_rmse = sqrt(mean_squared_error(y, predictions))

    return persistance_rmse_dict, total_rmse, predictions


def vecm_baseline_model(trainX, valX, column_names):
    """
    Fits a Vector Error Correction Model (VECM) to the training data and makes predictions on the validation data.

    Parameters:
    train_scaled (ndarray): The training dataset as a NumPy array.
    valid_scaled (ndarray): The validation dataset as a NumPy array.
    column_names (list): The names of the columns in the dataset.

    Returns:
    Tuple[dict, float]: A tuple containing a dictionary with the RMSE for each feature and the total RMSE.
    """

    # Flatten the data back into 2D
    trainX = trainX.reshape(trainX.shape[0], -1)
    valX = valX.reshape(valX.shape[0], -1)

    model = VECM(endog=trainX)
    model_fit = model.fit()

    vecm_valid_predictions = model_fit.predict(steps=len(valX))

    vecm_rmse = np.sqrt(mean_squared_error(
        valX, vecm_valid_predictions, multioutput='raw_values'))
    total_vecm_rmse = np.sqrt(mean_squared_error(valX, vecm_valid_predictions))

    vecm_rmse_dict = dict(zip(column_names, vecm_rmse))

    return vecm_rmse_dict, total_vecm_rmse, vecm_valid_predictions
