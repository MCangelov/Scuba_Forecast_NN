import pandas as pd
from sqlalchemy import create_engine


def get_beach_data(database_connector):
    beaches_lat_lon_info = pd.read_csv('csv_data/beach_info.csv', index_col=0)
    while True:
        try:
            index = int(input("Enter the index of the beach (0 to {}): ".format(
                len(beaches_lat_lon_info) - 1)))
            beach_name_sql_table = beaches_lat_lon_info.iloc[index][0].replace(
                ' ', '_').lower()
            print("\nSelected Beach Details:")
            print(beach_name_sql_table)
            break
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid index.")

    engine = create_engine(database_connector)
    single_beach_data = pd.read_sql_table(
        beach_name_sql_table, engine, index_col="datetime", parse_dates=["datetime"])
    engine.dispose()

    return single_beach_data, beach_name_sql_table
