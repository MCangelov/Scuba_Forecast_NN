from functions.data_load_and_transform.json_and_database import process_json_to_sql
from functions.data_load_and_transform.sql_connections import get_database_connector
from configparser import ConfigParser
import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


# Convert the extracted json files to the sql database. Filepaths are stored in beach_data_json_dir. JSON produced from black_sea_waves_reanalysis_to_json.py
config = ConfigParser()
config.read('config/beach_dir_config.ini')  # Hidden
beach_data_json_dir = config['beach_dir_config']['dir']

database_connector = get_database_connector()
process_json_to_sql(database_connector, beach_data_json_dir)
