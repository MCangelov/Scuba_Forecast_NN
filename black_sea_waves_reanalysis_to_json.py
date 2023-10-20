from functions.utils.misc import remove_months, add_months, beach_coordinates_locator
from functions.API_preprocessing.wave_feature_output import filter_beach_data
from functions.API_preprocessing.get_api import call_api
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
from typing import Tuple, Optional
import calendar
import getpass
import os
project_root = os.path.dirname(os.path.abspath(__file__))


USERNAME = input('Enter your username: ')
PASSWORD = getpass.getpass('Enter your password: ')
YYYY_MM_datetime_start_date = input(
    'Enter the extraction start date (YYYY-MM): ').split('-')
YYYY_datetime_end_date, MM_datetime_end_date = map(int, input(
    'Enter the extraction end date (YYYY-MM): ').split('-'))
OUTPUT_FILENAME = 'black_sea_waves_reanalysis.nc'
BEACH_INFO = pd.read_csv(os.path.join(
    project_root, 'csv_data', 'beach_info.csv'), index_col=0)


# Start_date always begin on the first day/00:00:00 of a month and end_date always ends on the last day/23:00:00 of a month.
start_date = f'{YYYY_MM_datetime_start_date[0]}-{YYYY_MM_datetime_start_date[1]}-01 00:00:00'
last_day_of_month = calendar.monthrange(
    YYYY_datetime_end_date, MM_datetime_end_date)[1]
end_date = f'{YYYY_datetime_end_date}-{MM_datetime_end_date:02d}-{last_day_of_month:02d} 23:00:00'

# Handle case where the end date is before the start date
if datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') > datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S'):
    raise Exception("The end date must be after the start date.")


# There is a limit to the size of the data that can be called (1gb), so we limit the data to 1 month.
# If a bigger range (hance bigger size) is put in, the data is batched, and time frames are processed separately.
total_month_range = None
datetime_start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
datetime_end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

delta = relativedelta(datetime_end_date, datetime_start_date)
total_month_range = delta.years * 12 + delta.months

if delta.days > 0 and datetime_end_date.day < datetime_start_date.day:  # Leap years and varying month days
    total_month_range -= 1


# Used to specify start/end date for beach data extraction
def range_defined_beach_data(start_date: str, end_date: str, total_month_range: int) -> pd.DataFrame:

    def get_monthly_beach_data(start_date: str, end_date: str, beach_coordinates: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches and filters beach data for a given date range.

        This function calls an API to get unfiltered beach data for a specified date range. 
        It then filters this data based on the provided beach coordinates. 
        If no coordinates are provided, it uses the `beach_coordinates_locator` function 
        to determine the coordinates.

        Parameters:
        start_date (str): The start date for fetching the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching the data in 'YYYY-MM-DD' format.
        beach_coordinates (tuple, optional): A tuple containing the latitude and longitude of the beach. 
                                            Defaults to None.

        Returns:
        tuple: A tuple where the first element is the filtered beach data and the second element 
            is the beach coordinates used for filtering.
        """

        unfiltered_beach_data = call_api(
            USERNAME, PASSWORD, OUTPUT_FILENAME, start_date, end_date)
        if beach_coordinates == None:
            beach_coordinates = beach_coordinates_locator(
                BEACH_INFO, unfiltered_beach_data)
        return filter_beach_data(beach_coordinates, unfiltered_beach_data), beach_coordinates

    filtered_monthly_beach_data = []

    if total_month_range == 0:  # Edge case where only one month is processed
        filtered_monthly_beach_data, beach_coordinates = get_monthly_beach_data(
            start_date, end_date, beach_coordinates=None)
    else:
        # We carry out this trick, in order to account for varying month length (i.e. 30 or 31 days) and leap years
        for i in range(total_month_range + 1):
            if i == 0:
                # The start_date & end_date are set to the same month
                end_date = remove_months(end_date, total_month_range)
                filtered_monthly_beach_data, beach_coordinates = get_monthly_beach_data(
                    start_date, end_date, beach_coordinates=None)
            else:
                # With each iteration, shift start and end by 1 month
                start_date = add_months(start_date, 1)
                # The add_months function accounts for the month length variability
                end_date = add_months(end_date, 1)

                one_month_filtered_beach_data, beach_coordinates = get_monthly_beach_data(
                    start_date, end_date)

                if len(one_month_filtered_beach_data) != 63:
                    raise ValueError("Missing beach data")

                filtered_monthly_beach_data = [pd.concat([df1, df2], ignore_index=False)  # type: ignore
                                               for df1, df2 in zip(filtered_monthly_beach_data, one_month_filtered_beach_data)]

    beach_df = pd.concat(filtered_monthly_beach_data,  # type: ignore
                         keys=beach_coordinates['beach_name'])  # type: ignore
    beach_df.index.set_names(['beach_name', 'time'], inplace=True)
    return beach_df


# Data for the specified time range
temporal_beach_data_df = range_defined_beach_data(
    start_date, end_date, total_month_range)


filename = f'temporal_beach_data_df_{datetime_start_date.year}_{datetime_start_date.month}-{datetime_end_date.year}_{datetime_end_date.month}.json'
temporal_beach_data_df.to_json(filename, orient='split')
