# -*- coding: utf-8 -*-
"""
Created on 5/18/2020 8:15 PM

@author: bill-
"""
import time
from datetime import datetime as dt
import numpy as np
import pandas as pd
from alpha_vantage.techindicators import TechIndicators

from Algo_trader_V1.api import Python_alpaca_API_connector as api
from Algo_trader_V1.live_model_functions.Python_AV_get_intraday_stock import pull_intraday_data, pull_stock_data, submit_order

intra_df = pull_intraday_data(symbol='TSLA',
                              interval='5min',
                              outputsize='full',
                              output_format='pandas',
                              plot_price=False)
# intra_df['open_diff'] = intra_df['open'].diff()
# intra_df = pred_feat(df=intra_df)

# monthly data
stock_df = pull_stock_data(symbol='NVDA',
                           adjusted=True,
                           outputsize='full',
                           cadence='monthly',
                           output_format='pandas',
                           plot_price=False)
# stock_df['open_diff'] = stock_df['open'].diff()
# adds features and removes NaNs
# stock_df = pred_feat(df=stock_df)


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


# time loop for trading logic
def simple_loop():

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

# loop based on the WEIGHTED MOVING AVERAGE
# this loop does not allow shorting

def wma_loop(stock_symbol):
    """
    symbol : 'XXXX'
    interval : 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    time_period : time_period=60, time_period=200
    series_type : close, open, high, low
    datatype : 'json', 'csv', 'pandas'
    """
    last_price = pull_intraday_data(symbol=stock_symbol,
                                    interval='5min',
                                    outputsize='full',
                                    output_format='pandas')
    # calculate the mean price of the last 25 min of the trading day
    mean_price = last_price['open'][:5].mean()
    # retrieve the very last quote to compare with
    actual_price = last_price['open'][:1].mean()

    # retrieve accounts remaining buying power
    bp = float(api.get_account().buying_power)

    # tech indicator returns a tuple; sma dictionary with values; meta dict with characteristics
    # instantiate the class first and provide the API key
    print("Retrieving weighted moving averages...")
    ti = TechIndicators('PKS7JXWMMDQQXQNDWT2P')
    wma_50, meta_wma_50 = ti.get_wma(symbol=stock_symbol, interval='daily', time_period='50', series_type='open')
    wma_200, meta_wma_200 = ti.get_wma(symbol=stock_symbol, interval='daily', time_period='200', series_type='open')

    while True:
        '''
        ACCESS OF WEIGHTED MOVING AVERAGES AND CONSECUTIVE INTERSECTION THEREOF
        naming of day + 1 is inverted to index position because list is in descending order
        then the increasing index element in the list represents a day
        further in the past (smaller date number)
        '''
        # zero indexed counter with values selected before index 3(last element exclusive); start at index 0

        key_list = sorted(enumerate(wma_50.keys()), reverse=False)[:3]
        key_list_2 = sorted(enumerate(wma_200.keys()), reverse=False)[:3]
        # access tuples inside list with key_list[LIST_INDEX][TUPLE_ELEMENT] (both 0-indexed)
        # comparison loop
        if (wma_50[key_list[2][1]]['WMA'] < wma_200[key_list_2[2][1]]['WMA'] and
                wma_50[key_list[0][1]]['WMA'] > wma_200[key_list_2[0][1]]['WMA']):
            # buy signal
            print("Executing buy signal...")
            try:
                print("Stock is being purchased")
                submit_order(symbol=stock_symbol,
                             qty=bp,
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol=stock_symbol,
                             qty=5,
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
            print(f"{symbol} is being bought")

        # check if wma_50 is smaller than wma_200; the stock is owned; at least one stock is owned
        elif (wma_50[key_list[2][1]]['WMA'] > wma_200[key_list_2[2][1]]['WMA'] and
                wma_50[key_list[0][1]]['WMA'] < wma_200[key_list_2[0][1]]['WMA']) and\
                (symbol in portfolio_list and portfolio_list[1] > 0):
            # execute sell signal
            print("Executing sell signal...")
            try:
                print("Stock is owned and is being sold...")
                submit_order(symbol=stock_symbol,
                             qty=2,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol=stock_symbol,
                             qty=float(stock_df['high'].head(1) / bp * 0.1),
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
                pass
        else:
            print("No action conducted at", dt.now().isoformat())
            time.sleep(5)


def ma_loop(stock_symbol):
    """
    Parameters
    -----------------
    symbol : 'XXXX'
    interval : 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    time_period : time_period=60, time_period=200
    series_type : close, open, high, low
    datatype : 'json', 'csv', 'pandas'
    """

    while True:

        '''endless loop for buying and selling'''
        # create a iterable tuple for the orders;
        start_time = time.time()
        last_price = pull_intraday_data(symbol=stock_symbol,
                                        interval='5min',
                                        outputsize='full',
                                        output_format='pandas')
        # retrieve the very last quote to compare with
        actual_price = last_price['open'][:1].mean()
        # retrieve accounts remaining buying power
        bp = float(api.get_account().buying_power)

        pos = Python_alpaca_API_connector.list_positions()
        portfolio_list = []
        print("Current portfolio positions:\n SYMBOL | NO. STOCKS")
        for i in range(0, len(pos), 1):
            # print as tuple
            print((pos[i].symbol, pos[i].qty))
            # append a tuple with the stock and quantity held
            portfolio_list.append((pos[i].symbol, pos[i].qty))

        '''
        ACCESS OF WEIGHTED MOVING AVERAGES AND CONSECUTIVE INTERSECTION THEREOF
        naming of day + 1 is inverted to index position because list is in descending order
        then the increasing index element in the list represents a day
        further in the past (smaller date number)
        '''
        # zero indexed counter with values selected before index 3(last element exclusive); start at index 0
        # tech indicator returns a tuple; sma dictionary with values; meta dict with characteristics
        # instantiate the class first and provide the API key
        print("Retrieving moving averages...")
        ti = TechIndicators('PKS7JXWMMDQQXQNDWT2P')
        sma_50, meta_sma_50 = ti.get_sma(symbol=stock_symbol, interval='daily', time_period='50', series_type='open')
        sma_200, meta_sma_200 = ti.get_sma(symbol=stock_symbol, interval='daily', time_period='200', series_type='open')

        key_list = sorted(enumerate(sma_50.keys()), reverse=False)[:3]
        key_list_2 = sorted(enumerate(sma_200.keys()), reverse=False)[:3]
        # access tuples inside list with key_list[LIST_INDEX][TUPLE_ELEMENT] (both 0-indexed)
        # comparison loop
        # check if sma_50 is intersecting sma_200 coming from below
        if (sma_50[key_list[2][1]]['SMA'] < sma_200[key_list_2[2][1]]['SMA'] and
                sma_50[key_list[0][1]]['SMA'] > sma_200[key_list_2[0][1]]['SMA']):
            # buy signal
            print("Executing buy signal...")
            try:
                print(f"Stock {stock_symbol} is being purchased")
                submit_order(symbol=stock_symbol,
                             qty=2,
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
                print("script execution time:", time.time() - start_time, "sec.")
            except BaseException as e:
                print(e)
                submit_order(symbol=stock_symbol,
                             qty=float(stock_df['high'].head(1) / bp * 0.1),
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
            print(f"{stock_symbol} is being bought")

        # check if sma_50 is intersecting sma_200 coming from above; the stock is owned; at least one stock is owned
        elif (sma_50[key_list[2][1]]['SMA'] > sma_200[key_list_2[2][1]]['SMA'] and
                sma_50[key_list[0][1]]['SMA'] < sma_200[key_list_2[0][1]]['SMA']) and\
                (symbol in portfolio_list and portfolio_list[1] > 0):
            # sell signal
            print("Executing sell signal...")
            print(f"{stock_symbol} is being sold")
            try:
                print(f"Stock {stock_symbol} is being sold")
                submit_order(symbol=stock_symbol,
                             qty=2,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol=stock_symbol,
                             qty=3,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
                pass
            print("script execution time:", time.time() - start_time, "sec.")
        else:
            print("No action needed to be conducted at", dt.now().isoformat())
            time.sleep(5)
