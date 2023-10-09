import pandas as pd

def wave_feature_output(beach_info_sensor, bs_df):
    result_df = []
    for beach_i in range(len(beach_info_sensor)):
        var_dict = {}

        for time in bs_df.index.levels[0]:
            lat_point = beach_info_sensor.iloc[beach_i].lat_sensor[1]
            lon_point = bs_df.loc[(time, lat_point)].index[0]

            if lat_point in bs_df.index.levels[1]:
                pass   
            else:        
                raise Exception('Latitude sensor not found')

            if lon_point in bs_df.loc[(time, lat_point)].index:
                pass
            else:
                raise Exception('Longitude Sensor not found')

            var_dict[time] = bs_df.loc[(time, lat_point, lon_point)]

        e = pd.DataFrame.from_dict(var_dict, orient = 'index')
        result_df.append(e)
    return result_df