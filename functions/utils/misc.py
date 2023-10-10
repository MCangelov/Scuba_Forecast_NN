from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar

def remove_months(end_date_str, months_to_remove):
    """
    Removes the specified number of months from the given end date.

    Args:
        end_date_str (str): The end date in the format "%Y-%m-%d %H:%M:%S".
        months_to_remove (int): The number of months to remove.

    Returns:
        str: The new end date after removing the specified number of months.
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    end_date = datetime.strptime(end_date_str, date_format)

    new_end_date = end_date - relativedelta(months=months_to_remove)

    return new_end_date.strftime(date_format)



def add_months(start_date_str, months_to_remove):
    """
    Adds the specified number of months from the given end date.

    Args:
        start_date_str (str): The start date in the format '%Y-%m-%d %H:%M:%S'.
        months_to_remove (int): The number of months to add.

    Returns:
        str: The new start date, with the day set to the last day of the month if necessary.
    """
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)

    new_start_date = start_date + relativedelta(months=months_to_remove)

    # Necessary to get the last day of the month, due to it being dynamic.
    if new_start_date.day != 1:
        last_day_of_month = calendar.monthrange(
            new_start_date.year, new_start_date.month)[1]
        new_start_date = new_start_date.replace(day=last_day_of_month)
        return new_start_date.strftime(date_format)

    return new_start_date.strftime(date_format)


# Binary search
def binary_search(arr, target):
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


def sensor_locator(beach_info, bs_df):
    """
    Assigns lat_sensor and lon_sensor values to each row in beach_info based on binary search in bs_df.

    Args:
        beach_info (pandas.DataFrame): DataFrame containing beach information.
        bs_df (pandas.DataFrame): DataFrame for binary search.

    Returns:
        pandas.DataFrame: Updated beach_info DataFrame with lat_sensor and lon_sensor values.
    """
    beach_info['lat_sensor'] = None
    beach_info['lon_sensor'] = None

    for i, lat_val in enumerate(beach_info['latitude']):
        c = binary_search(bs_df.index.levels[1], lat_val)
        beach_info['lat_sensor'][i] = c

    for i, lon_val in enumerate(beach_info['longitude']):
        d = binary_search(bs_df.index.levels[2], lon_val)
        beach_info['lon_sensor'][i] = d
    return beach_info