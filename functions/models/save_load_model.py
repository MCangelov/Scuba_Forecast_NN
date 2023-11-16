import os
import glob
import pickle

from keras.models import model_from_json
from typing import Dict, Any


def save_models(models: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """
    Save Keras models' weights, training histories, and architectures to a local directory.

    Parameters:
    models (Dict[str, Dict[str, Any]]): The dictionary containing the Keras models and their histories.
    save_dir (str): The directory to save the model files to.
    """
    for model_name, model_info in models.items():
        model = model_info['model']
        history = model_info['history']

        # Save model weights
        model.save_weights(os.path.join(save_dir, f'{model_name}_weights.h5'))

        # Save model history
        with open(os.path.join(save_dir, f'{model_name}_history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

        # Save model architecture
        model_json = model.to_json()
        with open(os.path.join(save_dir, f'{model_name}_architecture.json'), 'w') as f:
            f.write(model_json)

    print(f'Models saved to {save_dir}')


def load_models(load_dir: str) -> None:
    """
    Load Keras models' weights, training histories, and architectures from a local directory and print the best values of the metrics.

    Parameters:
    load_dir (str): The directory to load the model files from.
    """
    # Get a list of all model names in the directory
    model_names = [os.path.basename(name).replace('_architecture.json', '')
                   for name in glob.glob(os.path.join(load_dir, '*_architecture.json'))]

    # Print the column names
    print(f"{'Model':<20} {'Dropout':<10} {'RMSE':<10} {'MAE':<10} {'Val RMSE':<10} {'Val MAE':<10}")

    for model_name in model_names:
        # Load model weights
        model = model_from_json(
            open(os.path.join(load_dir, f'{model_name}_architecture.json')).read())
        model.load_weights(os.path.join(  # type: ignore
            load_dir, f'{model_name}_weights.h5'))

        # Load model history
        with open(os.path.join(load_dir, f'{model_name}_history.pkl'), 'rb') as f:
            history = pickle.load(f)

        dropout_str = model_name.split(', dropout ')[-1]
        dropout = float(dropout_str)
        model_name = model_name.replace(', dropout ' + dropout_str, '')
        rmse = round(history['root_mean_squared_error'][-1], 5)
        mae = round(history['mean_absolute_error'][-1], 5)
        val_rmse = round(history['val_root_mean_squared_error'][-1], 5)
        val_mae = round(history['val_mean_absolute_error'][-1], 5)
        print(
            f"{model_name:<20} {dropout:<10} {rmse:<10} {mae:<10} {val_rmse:<10} {val_mae:<10}")
