"""
This module connects to the Alpha Vantage API and allows data queries
IMPORTANT LIMIT AV API 5 API requests per minute and 500 requests per day
pull_data_adj = pulls data in intervals of days or higher
pull_intraday_data = dataframes for minute intervals

returns a dataframe
"""

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

from Algo_trader_V1.api.alpaca_API_connector import api
from Algo_trader_V1.api.av_acc_config import AV_API_KEY


# TODO
# last price with changes
# ts.get_quote_endpoint('AAPl')

# pull stock data from Alpha Vantage; returned as tuple
def pull_data_adj(symbol, outputsize, cadence, output_format):

    """
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    outputsize: 'compact'(100 rows for faster calls) or 'Full'(full retrieval)
    cadence: 'daily' / 'weekly' / 'monthly'
    output_format: ['json', 'csv', 'pandas']
    """
    ts = TimeSeries(key=AV_API_KEY, output_format=output_format,
                    treat_info_as_error=True, indexing_type='date', proxy=None)
    data = pd.DataFrame()
    try:
        if cadence == 'daily':
            data, meta_data = ts.get_daily_adjusted(symbol=symbol,
                                                outputsize=outputsize)
        if cadence == 'weekly':
            data, meta_data = ts.get_weekly_adjusted(symbol=symbol)
        if cadence == 'monthly':
            data, meta_data = ts.get_monthly_adjusted(symbol=symbol)
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

    except BaseException as e:
        print("<<<AV-API Problem!>>>")
        print(e)
    return data

# pull intraday data; returned as tuple
def pull_intraday_data(symbol, interval, outputsize, output_format):

    """
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    interval: ['1min', '5min', '15min', '30min', '60min']
    outputsize: 'full'
    output_format: ['json', 'csv', 'pandas']
    plot_price: boolean; generate a plot with open price and trading volume
    """
    ts = TimeSeries(key=AV_API_KEY, output_format=output_format,
                    treat_info_as_error=True, indexing_type='date', proxy=None)
    data = pd.DataFrame()
    try:
        data, meta_data = ts.get_intraday(symbol=symbol,
                                        interval=interval,
                                        outputsize=outputsize)
        #drop the date as index to use it for plotting
        data = data.reset_index(drop=False, inplace=False)
        data = data.rename(columns={"1. open": "open",
                                    "2. high": "high",
                                    "3. low": "low",
                                    "4. close": "close",
                                    "5. volume": "volume"},
                                    inplace=False)
    except BaseException as e:
        print("<<<AV-API Problem>>>!")
        print(e)
    return data, meta_data


def submit_order(symbol, qty, side, order_type, time_in_force, limit_price):

    """
     DOCUMENTATION
     symbol: Abbr in 'XXX',
     qty: int,
     side: 'buy' / 'sell',
     type: 'limit',
     time_in_force: 'gtc' / 'day',
     limit_price: Any = fl32 (with or without ''),
     stop_price: LIMIT ORDERS DO NOT REQUIRE A STOP PRICE
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
        asset_list = api.list_assets(status=status,
                        asset_class=asset_class
                        )
    except BaseException as f:
        print(f)
    return asset_list


