import json 
import pandas as pd
import os

from sqlalchemy import create_engine, MetaData
from contextlib import contextmanager
from pathlib import Path


def convert_to_multiindex(json_data):
    """
    Used by process_json_file to format the data into a Pandas DataFrame with a MultiIndex.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with MultiIndex (beach_name, datetime) and specified columns.
    """


    df = pd.DataFrame(json_data['data'], columns = json_data['columns'])
    df.index = pd.MultiIndex.from_tuples(json_data['index'], names=['beach_name', 'datetime'])
    df.reset_index(inplace = True)
    df['datetime'] = pd.to_datetime(df['datetime'], unit = 'ms') 
    df.set_index(['beach_name', 'datetime'], inplace = True)
    
    return df

def process_json_file(file_path):
    """
    Process a JSON file into a MultiIndex DataFrame and group it.

    Parameters:
    -----------
    file_path : str
        Path to the input JSON file.

    Returns:
    --------
    pandas.DataFrame, pandas.core.groupby.generic.DataFrameGroupBy
        - MultiIndex DataFrame.
        - Grouped DataFrame by the first MultiIndex level.

    Example:
    --------
    multi_index_df, grouped_df = process_json_file('data.json')
    """

    with open(file_path, 'r') as file:
        json_data = json.load(file)

    multi_index_df = convert_to_multiindex(json_data)
    grouped_df = multi_index_df.groupby(level=0, sort=False)
    
    return multi_index_df, grouped_df

def process_json_to_sql(db_url, beach_data_json_dir):
    @contextmanager
    def database_context(db_url):
        engine = create_engine(db_url)
        try:
            yield engine
        finally:
            engine.dispose()

    beach_data_json_filenames = os.listdir(beach_data_json_dir)
    beach_dir_filenames = [filename for filename in beach_data_json_filenames if filename.startswith('beach_df_') and filename.endswith('.json')]
    beach_dir_filenames.sort(key=lambda x: x.split('_')[1])
        

    beach_data_dir = Path(beach_data_json_dir)
    beach_data_path = [beach_data_dir / Path(filename) for filename in beach_dir_filenames]

    with database_context(db_url) as engine:
        metadata = MetaData()

        for i, filename in enumerate(beach_data_path):

            multiindex_df, _ = process_json_file(filename)

            mask_index = multiindex_df.index.duplicated()
            if mask_index.any():
                multiindex_df = multiindex_df[~mask_index]
                print(f'Check mask_index')

            multiindex_df.reset_index(inplace=True)

            for beach_name_sql, separate_beach_df in multiindex_df.groupby('beach_name', sort=False):
                table_name = beach_name_sql.replace(' ', '_').lower()
                metadata.reflect(bind=engine)
                separate_beach_df.drop(columns='beach_name', inplace=True)

                if table_name not in metadata.tables:
                    separate_beach_df.head(0).to_sql(table_name, engine, index=False)

                separate_beach_df.to_sql(table_name, engine, if_exists='append', index=False)