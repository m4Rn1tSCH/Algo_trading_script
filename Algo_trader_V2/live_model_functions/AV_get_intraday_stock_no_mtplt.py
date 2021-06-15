"""
This module connects to the Alpha Vantage API and allows data queries

IMPORTANT LIMIT AV API 5 API requests per minute and 500 requests per day

pull_data_adj = pulls data in intervals of days or higher
pull_intraday_data = dataframes for minute intervals
returns a dataframe
"""

import pandas as pd
# from decouple import config
# import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from Algo_trader_V2.api.alpaca_API_connector import api


# pull stock data from Alpha Vantage; returned as tuple
def pull_data_adj(symbol, outputsize, cadence, output_format):

    """
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    outputsize: 'compact'(100 rows for faster calls) or 'Full'(full retrieval)
    cadence: 'daily' / 'weekly' / 'monthly'
    output_format: ['json', 'csv', 'pandas']
    """
    ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format=output_format,
                    treat_info_as_error=True, indexing_type='date', proxy=None)
    if cadence == 'daily':
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
    if cadence == 'weekly':
        data, _ = ts.get_weekly_adjusted(symbol=symbol)
    if cadence == 'monthly':
        data, _ = ts.get_monthly_adjusted(symbol=symbol)
    # drop the date as index to use it
    data = data.reset_index(drop=False, inplace=False)
    # rename columns names for better handling
    data = data.rename(columns={"1. open": "open",
                                "2. high": "high",
                                "3. low": "low",
                                "4. close": "close",
                                "5. adjusted close": "adjusted_close",
                                "6. volume": "volume",
                                "7. dividend amount": "dividend amount"},
                                inplace=False)
    return data


# pull intraday data; returned as tuple
def pull_intraday_data(symbol, interval, outputsize, output_format):

    """
    Intraday_data returns a tuple with the pandas df and a meta data dictionary.
    The metadata has been dropped currently as it is not needed.
    symbol: pick abbreviation in letter strings 'XXXX'
    interval: ['1min', '5min', '15min', '30min', '60min']
    outputsize: 'full'
    output_format: ['json', 'csv', 'pandas']
    plot_price: boolean; generate a plot with open price and trading volume
    """
    ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format=output_format,
                    treat_info_as_error=True, indexing_type='date', proxy=None)
    data, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
    data = data.reset_index(drop=False, inplace=False)
    data = data.rename(columns={"1. open": "open",
                                "2. high": "high",
                                "3. low": "low",
                                "4. close": "close",
                                "5. volume": "volume"},
                                inplace=False)

    return data


def av_intraday(symbol):
    """
    pull ohlcv object and convert to pandas df
    :return: df with OHLCV columns
    """
    ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                    proxy=None)
    data, meta_data = ts.get_intraday(symbol=symbol, interval='1min', outputsize='full')

    return data, meta_data


def av_daily_adj(symbol):
    """
    pull daily adjusted
    :return:
    """
    ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                    proxy=None)
    data, meta_data = ts.get_daily_adjusted(symbol=symbol, interval='1min', outputsize='full')

    return data, meta_data


def submit_order(symbol, qty, side, order_type, time_in_force, limit_price):

    """
     :param symbol: str; Abbr in 'XXX',
     :param qty: int,
     :param side: 'buy' / 'sell',
     :param order_type: 'limit',
     :param time_in_force: 'gtc' / 'day',
     :param limit_price: Any = fl32 (with or without ''),
     :param stop_price: LIMIT ORDERS DO NOT REQUIRE A STOP PRICE
     """
    try:
        api.submit_order(symbol=symbol,
                         qty=qty,
                         side=side,
                         type=order_type,
                         time_in_force=time_in_force,
                         limit_price=limit_price
                         )
    except BaseException as f:
        print(f)
    return 'Order submitted'

# list all equities
def get_asset_list(status, asset_class):

    """
    Generate a list of all assets
    status:
    asset_class:
    :return:
    list of asset in possession
    """
    try:
        asset_list = api.list_assets(status=status, asset_class=asset_class)
        print(asset_list)
    except BaseException as f:
        print(f)
    return
