import pandas as pd
from configparser import ConfigParser
from sqlalchemy import create_engine
from typing import Tuple


def get_database_connector() -> str:
    """
    Reads the database configuration from a file and returns a database connector string.

    Returns:
        str: A string that can be used to connect to PostgreSQL.
    """
    config = ConfigParser()
    config.read('config/postgres_config.ini')
    username = config['postgres_config']['username']
    password = config['postgres_config']['password']
    host = config['postgres_config']['host']
    port = config['postgres_config']['port']
    database_name = config['postgres_config']['database_name']
    database_connector = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"

    return database_connector


def get_beach_data(database_connector: str) -> Tuple[pd.DataFrame, str]:
    """
    Fetches data for a specific beach from the database.

    Args:
        database_connector (str): The connector string for the database.

    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing a DataFrame with the beach data and the name of the beach.
    """
    beaches_lat_lon_info = pd.read_csv('csv_data/beach_info.csv', index_col=0)
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            index = int(input("Enter the index of the beach (0 to {}): ".format(
                len(beaches_lat_lon_info) - 1)))
            beach_name_sql_table = beaches_lat_lon_info.iloc[index][0].replace(
                ' ', '_').lower()
            print("\nSelected Beach Details:")
            print(beach_name_sql_table)
            break
        except (ValueError, IndexError):
            attempts += 1
            print("Invalid input. Please enter a valid index.")
            if attempts == max_attempts:
                print("Maximum number of attempts reached. Exiting function.")
                return None, None

    engine = create_engine(database_connector)
    single_beach_data = pd.read_sql_table(
        beach_name_sql_table, engine, index_col="datetime", parse_dates=["datetime"])
    engine.dispose()

    return single_beach_data, beach_name_sql_table
