# -*- coding: utf-8 -*-
"""
Created on 5/18/2020 8:15 PM

@author: bill-
"""
from datetime import datetime as dt
import pandas as pd
import numpy as np
import time

from alpha_vantage.techindicators import TechIndicators
from Python_AV_get_intraday_stock import pull_intraday_data, pull_stock_data, submit_order
from Python_prediction_features import pred_feat

"""
-pull data every hour /every day
-prepare df and decide what to sell or buy
if loop -> buy or sell
append data every hour
calculate/or mark -> sell or buy order
"""

# intra day data
intra_df = pull_intraday_data(symbol='TSLA',
                              interval='5min',
                              outputsize='full',
                              output_format='pandas')
intra_df['open_diff'] = intra_df['open'].diff()
intra_df = pred_feat(df=intra_df)
print(intra_df.head(3))

# monthly data
stock_df = pull_stock_data(symbol='NVDA',
                           adjusted=True,
                           outputsize='full',
                           cadence='monthly',
                           output_format='pandas')
stock_df['open_diff'] = stock_df['open'].diff()
# adds features and removes NaNs
stock_df = pred_feat(df=stock_df)
print(stock_df.head(3))


def trading_support_resistance(df, bin_width=30):


    """
    create empty indicator columns and add values as the data evovles
    sup_tol : supportive tolerance
    res_tol : resistance tolerance
    sup_count : support count
    res_count : resistance count
    sup : support
    res : resistance
    positions : positions open
    signal : signals found
    """

    df['sup_tol'] = pd.Series(np.zeros(len(df)))
    df['res_tol'] = pd.Series(np.zeros(len(df)))
    df['sup_count'] = pd.Series(np.zeros(len(df)))
    df['res_count'] = pd.Series(np.zeros(len(df)))
    df['sup'] = pd.Series(np.zeros(len(df)))
    df['res'] = pd.Series(np.zeros(len(df)))
    df['positions'] = pd.Series(np.zeros(len(df)))
    df['signal'] = pd.Series(np.zeros(len(df)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(df)):
        df_section = df[x - bin_width:x + 1]

    support_level = min(df_section['open'])
    resistance_level = max(df_section['open'])
    range_level = resistance_level - support_level

    df['res'][x] = resistance_level
    df['sup'][x] = support_level
    # allow a 20% buffer back into the mean zone of the price movement
    df['sup_tol'][x] = support_level + 0.2 * range_level
    # allow a 20% buffer back into the mean zone of the price movement
    df['res_tol'][x] = resistance_level - 0.2 * range_level

    if df['open'][x] >= df['res_tol'][x] and df['open'][x] <= df['res'][x]:
        in_resistance += 1
        df['res_count'][x] = in_resistance
    elif df['open'][x] <= df['sup_tol'][x] and df['open'][x] >= df['sup'][x]:
        in_support += 1
        df['sup_count'][x] = in_support
    else:
        in_support = 0
        in_resistance = 0

    if in_resistance > 2:
        df['signal'][x] = 1
    elif in_support > 2:
        df['signal'][x] = 0
    else:
        df['signal'][x] = df['signal'][x - 1]
        df['positions'] = df['signal'].diff()

    # produces NaNs in the df again!
    trading_support_resistance(df=stock_df, bin_width=30)

#time loop for trading logic
def test_loop():
    """
    Thank you for using Alpha Vantage!
    Our standard API call frequency is 5 calls per minute and 500 calls per day.
    Premium: https://www.alphavantage.co/premium/
    """
    while True:
        last_price = pull_intraday_data(symbol='TSLA',
                                      interval='5min',
                                      outputsize='full',
                                      output_format='pandas')
        # calculate the mean price of the last 25 min of the trading day
        mean_price = last_price['open'][:5].mean()
        # retrieve the very last quote to compare with
        actual_price = last_price['open'][:1].mean()
        print("Price retrieved; procuring stocks")
        if mean_price > actual_price:
            # buy signal
            try:
                print("Stock is being purchased")
                submit_order(symbol='TSLA',
                                 qty=2,
                                 side='buy',
                                 type='limit',
                                 time_in_force='gtc',
                                 limit_price=mean_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol='TSLA',
                             qty=2,
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
        elif mean_price < actual_price:
            try:
                print("Stock is being sold")
                submit_order(symbol='TSLA',
                             qty=2,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol='TSLA',
                             qty=2,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
        else:
            print("Both prices identical; no action")
        # loop will pause for x seconds
        time.sleep(600)


def wma_loop(symbol):

    """
    symbol : 'XXXX'
    interval : 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    time_period : time_period=60, time_period=200
    series_type : close, open, high, low
    datatype : 'json', 'csv', 'pandas'
    """
    while True:
        # tech indicator returns a tuple; sma dictionary with values; meta dict with characteristics
        # instantiate the class first and provide the API key
        ti = TechIndicators('PKS7JXWMMDQQXQNDWT2P')
        wma_50, meta_wma_50 = ti.get_wma(symbol='TSLA', interval='daily', time_period='50', series_type='open')
        wma_200, meta_wma_200 = ti.get_wma(symbol='TSLA', interval='daily', time_period='200', series_type='open')

        # TODO
        # fix access code for dictionary
        # dict: (key): ((inner key, inner value))
        for key, nested_value in wma_50.items():
            for sma_key, value in nested_value.items():
                print(value)
        # comparison loop
        if wma_50 > wma_200:
            # buy signal
            print(f"{symbol} is being bought")
        elif wma_50 < wma_200 and # stock owned:
            # sell signal
            print(f"{symbol} is being bought")
        else:
            break
            time.sleep(600)