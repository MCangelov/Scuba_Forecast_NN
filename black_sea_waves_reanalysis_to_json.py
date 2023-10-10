import calendar
import getpass

import pandas as pd

import sys
sys.path.append('C:\\Users\\24\\Desktop\\Scraper\\scuba_scrapper\\functions')

from datetime import datetime
from dateutil.relativedelta import relativedelta

from functions.utils.misc import remove_months, add_months, sensor_locator
from functions.API_preprocessing.wave_feature_output import wave_feature_output
from functions.API_preprocessing.get_api import get_data


USERNAME = input('Enter your username: ')
PASSWORD = getpass.getpass('Enter your password: ')

# date_start always begin on the first day/00:00:00 of a month and date_end always ends on the last day/23:00:00 of a month.
YYYY_MM_start = input('Enter the extraction start date (YYYY-MM): ').split('-')
date_start = f'{YYYY_MM_start[0]}-{YYYY_MM_start[1]}-01 00:00:00'

YYYY_end, MM_end = map(int, input(
    'Enter the extraction end date (YYYY-MM): ').split('-'))
LAST_DAY = calendar.monthrange(YYYY_end, MM_end)[1]
date_end = f'{YYYY_end}-{MM_end:02d}-{LAST_DAY:02d} 23:00:00'

# Handle case where the end date is before the start date
if datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S') > datetime.strptime(date_end, '%Y-%m-%d %H:%M:%S'):
    raise Exception("The end date must be after the start date.")

OUTPUT_FILENAME = 'black_sea_waves_reanalysis.nc'


# There is a limit to the size of the data that can be called, so we limit the data to 1 month.
# If a bigger range is put in, the data is batched, and time frames are processed separately.
BATCH_SIZE = None
START_DATE = datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(date_end, '%Y-%m-%d %H:%M:%S')

delta = relativedelta(END_DATE, START_DATE)
BATCH_SIZE = delta.years * 12 + delta.months

if delta.days > 0 and END_DATE.day < START_DATE.day:  # Leap years and varying month days
    BATCH_SIZE -= 1

filename = f'beach_df_{START_DATE.year}_{START_DATE.month}-{END_DATE.year}_{END_DATE.month}.json'

beach_info = pd.read_csv(
    'C:/Users/24/Desktop/Scraper/scuba_scrapper/beach_info.csv', index_col=0)

if BATCH_SIZE == 0:
    bs_df = get_data(USERNAME, PASSWORD, OUTPUT_FILENAME, date_start, date_end)
    beach_info_sensor = sensor_locator(beach_info, bs_df)
    result_df = wave_feature_output(beach_info_sensor, bs_df)

    beach_df = pd.concat(result_df, keys=beach_info_sensor['beach_name'])
    beach_df.index.set_names(['beach_name', 'time'], inplace=True)


else:
    for i in range(BATCH_SIZE + 1):
        # For the first iteration, we subtract the end month to make it the same as the start month
        if i == 0:
            date_end = remove_months(date_end, BATCH_SIZE)

            bs_df = get_data(USERNAME, PASSWORD,
                             OUTPUT_FILENAME, date_start, date_end)

            beach_info_sensor = sensor_locator(beach_info, bs_df)

            result_df = wave_feature_output(beach_info_sensor, bs_df)

        # Each consecutive iteration, shift both months by 1
        else:
            date_start = add_months(date_start, 1)
            date_end = add_months(date_end, 1)

            bs_df = get_data(USERNAME, PASSWORD,
                             OUTPUT_FILENAME, date_start, date_end)

            single_result_df = wave_feature_output(beach_info_sensor, bs_df)

            # Makes sure all beaches are present
            if len(single_result_df) != 63:
                raise ValueError("Missing beach data")

            result_df = [pd.concat([df1, df2], ignore_index=False)
                         for df1, df2 in zip(result_df, single_result_df)]
            beach_df = pd.concat(
                result_df, keys=beach_info_sensor['beach_name'])

beach_df.to_json(filename, orient = 'split')