import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import Tuple, Optional
from IPython.display import display, clear_output
import ipywidgets as widgets


def plot_interactive(df: DataFrame, timescale: str, date_range: Tuple[str, str], column_name: str, plot_type: str, num_lags: Optional[int] = None) -> None:
    """
    Plots the data in an interactive way based on the provided parameters.

    Args:
        df (DataFrame): The DataFrame containing the data to be plotted.
        timescale (str): The timescale for resampling the data. Options are 'Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'.
        date_range (Tuple[str, str]): A tuple containing the start and end dates for the data to be plotted.
        column_name (str): The name of the column in df to be plotted.
        plot_type (str): The type of plot to be generated. Options are 'Data', 'ACF/PACF', 'Lag Plot'.
        num_lags (int, Optional): The number of lags to be used if plot_type is 'Lag Plot'. Defaults to None.
    """

    single_beach_data_daily = df.resample('D').mean()
    single_beach_data_weekly = df.resample('W').mean()
    single_beach_data_monthly = df.resample('M').mean()
    single_beach_data_yearly = df.resample('Y').mean()

    if timescale == 'Hourly':
        data = df
    elif timescale == 'Daily':
        data = single_beach_data_daily
    elif timescale == 'Weekly':
        data = single_beach_data_weekly
    elif timescale == 'Monthly':
        data = single_beach_data_monthly
    elif timescale == 'Yearly':
        data = single_beach_data_yearly

    data = data.loc[date_range[0]:date_range[1]]  # type: ignore

    plt.close('all')
    if plot_type == 'Data':
        plt.figure(figsize=(15, 5))
        plt.plot(data.index, data[column_name])
        plt.title(f'{column_name} ({timescale} Resampled Data)')
        plt.xlabel('datetime')
        plt.ylabel(column_name)
        plt.xticks(rotation=45)
        plt.show()

    elif plot_type == 'ACF/PACF':
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        plot_acf(data[column_name], zero=False, lags=40, ax=ax1)
        ax1.set_title(f'ACF for {column_name} ({timescale})')
        plot_pacf(data[column_name], zero=False, lags=40, ax=ax2)
        ax2.set_title(f'PACF for {column_name} ({timescale})')
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    elif plot_type == 'Lag Plot':
        plt.figure()
        pd.plotting.lag_plot(data[column_name], lag=num_lags)  # type: ignore
        plt.title(f'Lag Plot for {column_name} ({timescale})')
        plt.show()


def create_widgets_and_plot(single_beach_data: DataFrame) -> widgets.VBox:
    """
    Creates interactive widgets for plotting data and handles their events.

    Args:
        single_beach_data (DataFrame): The DataFrame containing the data to be plotted.

    Returns:
        widgets.VBox: A VBox widget containing all the created widgets.
    """
    timescale_widget = widgets.Dropdown(
        options=['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'],
        value='Daily',
        description='Select Time Scale:'
    )

    plot_type_widget = widgets.Dropdown(
        options=['Data', 'ACF/PACF', 'Lag Plot'],
        value='Data',
        description='Select Plot Type:'
    )

    num_lags_widget = widgets.IntText(
        value=1,
        description='Number of Lags:',
        disabled=True
    )

    start_date_widget = widgets.DatePicker(
        description='Start Date',
        value=pd.to_datetime('1979-01-01')
    )

    end_date_widget = widgets.DatePicker(
        description='End Date',
        value=pd.to_datetime('2021-12-31')
    )

    column_name_widget = widgets.Dropdown(
        options=single_beach_data.columns.tolist(),
        value=single_beach_data.columns.tolist()[0],
        description='Select Column:'
    )

    button = widgets.Button(description="Generate Plot")

    def on_plot_type_change(change):
        if change['new'] == 'Lag Plot':
            num_lags_widget.disabled = False
        else:
            num_lags_widget.disabled = True

    plot_type_widget.observe(
        on_plot_type_change, names='value')  # type: ignore

    button = widgets.Button(description="Generate Plot")

    def update_plot(button):
        timescale = timescale_widget.value
        start_date = start_date_widget.value
        end_date = end_date_widget.value
        column_name = column_name_widget.value
        plot_type = plot_type_widget.value
        num_lags = num_lags_widget.value if not num_lags_widget.disabled else None

        date_range = (start_date, end_date)

        clear_output(wait=True)

        display(widgets.VBox([
            timescale_widget,
            widgets.HBox([start_date_widget, end_date_widget]),
            column_name_widget,
            plot_type_widget,
            num_lags_widget,
            button
        ]))

        plot_interactive(
            single_beach_data, timescale, date_range, column_name, plot_type, num_lags)  # type: ignore

    button.on_click(update_plot)

    return widgets.VBox([
        timescale_widget,
        widgets.HBox([start_date_widget, end_date_widget]),
        column_name_widget,
        plot_type_widget,
        num_lags_widget,
        button
    ])
