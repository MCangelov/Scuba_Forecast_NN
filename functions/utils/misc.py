from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List
import calendar
import pandas as pd
pd.options.mode.chained_assignment = None


def remove_months(end_date_str: str, months_to_remove: int) -> str:
    """
    Removes the specified number of months from the given end date and ensures that
    the new end date is the last day of the month.

    Args:
        end_date_str (str): The end date in the format "%Y-%m-%d %H:%M:%S".
        months_to_remove (int): The number of months to remove.

    Returns:
        str: The new end date after removing the specified number of months.
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    end_date = datetime.strptime(end_date_str, date_format)
    new_end_date = end_date - relativedelta(months=months_to_remove)

    # Necessary to get the last day of the month, due to it being dynamic.
    _, last_day_of_month = calendar.monthrange(
        new_end_date.year, new_end_date.month)
    new_end_date = new_end_date.replace(day=last_day_of_month)

    return new_end_date.strftime(date_format)


def add_months(start_date_str: str, months_to_add: int) -> str:
    """
    Adds the specified number of months to the given date.

    Args:
        start_date_str (str): The start date in the format '%Y-%m-%d %H:%M:%S'.
        months_to_add (int): The number of months to add.

    Returns:
        str: The new future date.
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)

    new_start_date = start_date + relativedelta(months=months_to_add)

    # Necessary to get the last day of the month, due to it being dynamic.
    if new_start_date.day != 1:
        last_day_of_month = calendar.monthrange(
            new_start_date.year, new_start_date.month)[1]
        new_start_date = new_start_date.replace(day=last_day_of_month)
        return new_start_date.strftime(date_format)

    return new_start_date.strftime(date_format)


def binary_search(arr: List[int], target: int) -> List[int]:
    """
    Perform binary search on a sorted array to find the target value.

    Args:
        arr (list): The sorted array.
        target (int): The target value to search for.

    Returns:
        list: A list containing the index and value of the target if found,
              otherwise the index and value of the closest element.
    """
    left, right = 0, len(arr) - 1
    closest_index = None

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return [mid, arr[mid]]

        if closest_index is None or abs(arr[mid] - target) < abs(arr[closest_index] - target):
            closest_index = mid

        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return [closest_index, arr[closest_index]]


def beach_coordinates_locator(beach_info: pd.DataFrame, beach_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns lat_sensor and lon_sensor values to each row in beach_info based on binary search in beach_df.

    Args:
        beach_info (pandas.DataFrame): DataFrame containing beach information.
        beach_df (pandas.DataFrame): DataFrame for binary search.

    Returns:
        pandas.DataFrame: Updated beach_info DataFrame with lat_sensor and lon_sensor values.
    """
    beach_info['lat_sensor'] = None
    beach_info['lon_sensor'] = None

    for i, lat_val in enumerate(beach_info['latitude']):
        c = binary_search(beach_df.index.levels[1], lat_val)
        beach_info['lat_sensor'][i] = c

    for i, lon_val in enumerate(beach_info['longitude']):
        d = binary_search(beach_df.index.levels[2], lon_val)

        beach_info['lon_sensor'][i] = d
    return beach_info
