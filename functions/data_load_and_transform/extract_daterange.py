def extract_daterange_df(df):
    """
    Get the data of the df based on user-input start and end dates.

    Enter start and end dates in 'YYYY-MM-DD HH' format,
    then filters the DataFrame to include only rows within that date range.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be filtered.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame if valid dates are provided.
    """

    start_input = input("Enter start date (YYYY-MM-DD HH): ")
    end_input = input("Enter end date (YYYY-MM-DD HH): ")

    start_input += '00:00'
    end_input += '00:00'

    start_datetime = pd.to_datetime(start_input, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    end_datetime = pd.to_datetime(end_input, format='%Y-%m-%d %H:%M:%S', errors='coerce')

    if start_datetime is not pd.NaT and end_datetime is not pd.NaT:
        filtered_df = df[(df.index >= start_datetime) & (df.index <= end_datetime)]
        return filtered_df
    else:
        print("Invalid date format. Please use the format 'YYYY-MM-DD'")
        return None