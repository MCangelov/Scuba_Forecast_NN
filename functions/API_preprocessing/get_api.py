# MotuOptions and motu_option_parser sourced from "https://help.marine.copernicus.eu/en/articles/5211063-how-to-use-the-motuclient-within-python-environment"

import motuclient
import xarray as xr
import pandas as pd


class MotuOptions:
    def __init__(self, attrs: dict):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None


def motu_option_parser(script_template, usr, pwd, output_filename, date_min, date_max):
    dictionary = dict(
        [e.strip().partition(" ")[::2] for e in script_template.split('--')])
    dictionary['variable'] = [value for (var, value) in [e.strip().partition(
        " ")[::2] for e in script_template.split('--')] if var == 'variable']
    for k, v in list(dictionary.items()):
        if v == '<OUTPUT_DIRECTORY>':
            dictionary[k] = '.'
        if v == '<OUTPUT_FILENAME>':
            dictionary[k] = output_filename
        if v == '<USERNAME>':
            dictionary[k] = usr
        if v == '<PASSWORD>':
            dictionary[k] = pwd
        if k == 'date-min':
            dictionary[k] = date_min
        if k == 'date-max':
            dictionary[k] = date_max
        if k in ['longitude-min', 'longitude-max', 'latitude-min',
                 'latitude-max', 'depth-min', 'depth-max']:
            dictionary[k] = float(v)
        dictionary[k.replace('-', '_')] = dictionary.pop(k)

    dictionary.pop('python')
    dictionary['auth_mode'] = 'cas'
    return dictionary


def call_api(USERNAME: str, PASSWORD: str, OUTPUT_FILENAME: str, DATE_START: str, DATE_END: str) -> pd.DataFrame:
    """
    Calls an API to fetch data, processes the data and returns a DataFrame.

    Parameters:
    USERNAME (str): The username for the API.
    PASSWORD (str): The password for the API.
    OUTPUT_FILENAME (str): The name of the output file where the fetched data will be stored.
    DATE_START (str): The start date for the data request in 'YYYY-MM-DD HH:MM:SS' format.
    DATE_END (str): The end date for the data request in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
    pd.DataFrame: A DataFrame containing the processed data.
    """

    API_request = 'python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu --service-id BLKSEA_MULTIYEAR_WAV_007_006-TDS --product-id cmems_mod_blk_wav_my_2.5km_PT1H-i --longitude-min 27.09038280355556 --longitude-max 28.605053299999998 --latitude-min 41.9582344 --latitude-max 43.742464399999996 --date-min "2021-12-01 00:00:00" --date-max "2021-12-31 23:00:00" --variable VHM0 --variable VHM0_SW1 --variable VHM0_SW2 --variable VHM0_WW --variable VMDR --variable VMDR_SW1 --variable VMDR_SW2 --variable VMDR_WW --variable VPED --variable VSDX --variable VSDY --variable VTM01_SW1 --variable VTM01_SW2 --variable VTM01_WW --variable VTM02 --variable VTM10 --variable VTMX --variable VTPK --variable VZMX --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'

    black_sea_data_request = motu_option_parser(
        API_request, USERNAME, PASSWORD, OUTPUT_FILENAME, DATE_START, DATE_END)
    motuclient.motu_api.execute_request(MotuOptions(black_sea_data_request))

    black_sea_dataset = xr.open_dataset(OUTPUT_FILENAME)
    black_sea_df = black_sea_dataset.to_dataframe()
    black_sea_df = black_sea_df.dropna(how='all')
    return black_sea_df
