import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import hvplot.pandas
from IPython.display import display, clear_output
import ipywidgets as widgets


def plot_interactive(df, timescale, date_range, column_name, plot_type):

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

    data = data.loc[date_range[0]:date_range[1]]

    if plot_type == 'Data':
        plot = data.hvplot.line('datetime', column_name, title=f'{column_name} ({timescale} Resampled Data)').opts(
            width=900, height=500, tools=['hover'], xrotation=45)

    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        plot_acf(data[column_name], zero=False, lags=40, ax=ax1)
        ax1.set_title(f'ACF for {column_name} ({timescale})')
        plot_pacf(data[column_name], zero=False, lags=40, ax=ax2)
        ax2.set_title(f'PACF for {column_name} ({timescale})')
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        plot = fig

    return plot


def create_widgets_and_plot(single_beach_data):
    timescale_widget = widgets.Dropdown(
        options=['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly'],
        value='Daily',
        description='Select Time Scale:'
    )

    plot_type_widget = widgets.Dropdown(
        options=['Data', 'ACF/PACF'],
        value='Data',
        description='Select Plot Type:'
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

    def update_plot(button):
        timescale = timescale_widget.value
        start_date = start_date_widget.value
        end_date = end_date_widget.value
        column_name = column_name_widget.value
        plot_type = plot_type_widget.value

        date_range = (start_date, end_date)

        plot = plot_interactive(
            single_beach_data, timescale, date_range, column_name, plot_type)

        clear_output(wait=True)

        # Explicitly display the widgets after the plot
        display(widgets.VBox([
            timescale_widget,
            widgets.HBox([start_date_widget, end_date_widget]),
            column_name_widget,
            plot_type_widget,
            button
        ]))

        display(plot)

    button.on_click(update_plot)

    return widgets.VBox([
        timescale_widget,
        widgets.HBox([start_date_widget, end_date_widget]),
        column_name_widget,
        plot_type_widget,
        button
    ])
