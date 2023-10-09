import pandas as pd

def check_missing_or_nan(single_beach_df, beach_name_sql_table):
    """
    Analyze a DataFrame for missing hours and NaN values.

    Parameters:
    -----------
    single_beach_df : pandas.DataFrame
        The DataFrame to be analyzed.

    beach_name_sql_table : str
        The name of the selected table.
    """
    index_range = pd.date_range("1979-01-01", "2021-12-31", freq='H')
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