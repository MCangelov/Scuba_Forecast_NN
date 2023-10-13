import pandas as pd
from typing import List


def filter_beach_data(beach_info_sensor: pd.DataFrame, unfiltered_beach_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters beach data based on sensor information and associated unfiltered beach data.

    Parameters:
        - beach_info_sensor (pd.DataFrame): A DataFrame containing sensor information.
        - unfiltered_beach_data (pd.DataFrame): A Multiindex DataFrame containing beach data.

    Returns:
        List[pd.DataFrame]: A list of DataFrames containing filtered beach data.
    """
    result_df = []
    for beach_i in range(len(beach_info_sensor)):
        var_dict = {}

        for time in unfiltered_beach_data.index.levels[0]:
            lat_point = beach_info_sensor.iloc[beach_i].lat_sensor[1]
            lon_point = unfiltered_beach_data.loc[(time, lat_point)].index[0]

            if lat_point in unfiltered_beach_data.index.levels[1]:
                pass
            else:
                raise Exception('Latitude sensor not found')

            if lon_point in unfiltered_beach_data.loc[(time, lat_point)].index:
                pass
            else:
                raise Exception('Longitude Sensor not found')

            var_dict[time] = unfiltered_beach_data.loc[(
                time, lat_point, lon_point)]

        e = pd.DataFrame.from_dict(var_dict, orient='index')
        result_df.append(e)
    return result_df
