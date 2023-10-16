import pandas as pd


def check_missing_or_nan(single_beach_df, beach_name_sql_table, DATA_STARTDATE="1979-01-01", DATA_ENDDATE="2021-12-31"):
    """
    Analyze a DataFrame for missing hours and NaN values.

    Parameters:
    -----------
    single_beach_df : pandas.DataFrame
        The DataFrame to be analyzed.

    beach_name_sql_table : str
        The name of the selected table.

    DATA_STARTDATE : str
        The start date of the dataset

    DATA_ENDDATE : str
        The end date of the dataset
    """
    index_range = pd.date_range(DATA_STARTDATE, DATA_ENDDATE, freq='H')
    missing_hours = index_range.difference(single_beach_df.index)

    if missing_hours.empty:
        print(f"No missing hours in {beach_name_sql_table}.")
    else:
        print("Missing hours:")
        print(missing_hours)

    if single_beach_df.isna().any().any():
        print(f"NaN values in {beach_name_sql_table}")
    else:
        print(f"No NaN in {beach_name_sql_table}")
