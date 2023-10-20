from typing import Tuple, Dict
from scipy.stats import shapiro, normaltest, anderson, kstest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def kpss_adf_stationarity(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Check the stationarity of a time series DataFrame.

    Calculates the p-values for Kwiatkowski-Phillips-Schmidt-Shin (KPSS) and
    Augmented Dickey-Fuller (ADF) stationarity tests 

    Parameters:
    -----------
    df : pandas.DataFrame
        The time series DataFrame to be analyzed.

    Returns:
    --------
    tuple of str
        A tuple indicating the stationarity of the time series based on ADF and KPSS tests.
        Each element can be 'Stationary' or 'Non-stationary'.
    """

    kps = kpss(df)
    adf = adfuller(df)

    kpss_pv, adf_pv = kps[1], adf[1]
    kpssh, adfh = 'KPSS - Stationary', 'ADF - Non-stationary'

    if kpss_pv < 0.05:
        kpssh = 'KPSS - Non-stationary (possible trend)'

    if adf_pv < 0.05:
        adfh = 'ADF - Stationary'

    return (kpssh, adfh)


# Imply a choice for method selection, or always include all
def check_stationarity(df: pd.DataFrame, non_stationary: dict) -> None:
    """
    Check which differencing method makes the series stationary.

    Parameters:
    -----------
    df : pandas.DataFrame
        The time series DataFrame to be analyzed.
    non_stationary : dict
        A dictionary of non-stationary columns.

    Returns:
    --------
    None
    """

    methods = {
        'first_order_diff': df.diff(),
        'second_order_diff': df.diff(52),
        'subtract_rolling_mean': df - df.rolling(window=52).mean(),
        'log_transform': np.log(df),
        'sd_detrend': seasonal_decompose(df).observed - seasonal_decompose(df).trend,
        'cyclic': hpfilter(df)[0]
    }

    for column in non_stationary.keys():
        print(f"\nColumn: {column}")
        for name, method in methods.items():
            method = method[column].dropna()
            kpss_s, adf_s = kpss_adf_stationarity(method)

            if kpss_s == 'KPSS - Stationary' and adf_s == 'ADF - Stationary':
                print(f'{name} --> KPSS: {kpss_s}, ADF: {adf_s}')


def normality_testing(df: pd.DataFrame, p_level: float = 0.05) -> Dict[str, str]:
    """
    Perform normality tests on a DataFrame.

    Calculates various normality tests (Shapiro-Wilk, D'Agostino and Pearson's,
    Anderson-Darling, Kolmogorov-Smirnov) and reports whether the data is 
    considered normal or non-normal based on a
    specified significance level (default is 0.05).

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be tested.

    p_level : float, optional
        The significance level for the normality tests (default is 0.05).

    Returns:
    --------
    dict
        A dictionary containing the results of each normality test
    """
    _, shapiro_pval = shapiro(df)
    _, normaltest_pval = normaltest(df)
    anderson_stat = anderson(df, dist='norm')
    _, kstest_pval = kstest(df, 'norm')

    results = {
        'Shapiro-Wilk': 'Pass normality' if shapiro_pval > p_level else 'Non-normality',
        "D'Agostino-Pearson": 'Pass normality' if normaltest_pval > p_level else 'Non-normality',
        'Anderson-Darling': 'Pass normality' if anderson_stat.statistic < anderson_stat.critical_values[2] else 'Non-normality',
        'Kolmogorov-Smirnov': 'Pass normality' if kstest_pval > p_level else 'Non-normality',
    }

    return results
