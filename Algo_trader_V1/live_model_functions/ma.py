# -*- coding: utf-8 -*-
"""
Created on 9/27/2020; 9:22 AM

@author: Bill Jaenke
"""
import time
from datetime import datetime as dt

from alpha_vantage.techindicators import TechIndicators

from Algo_trader_V1.api.alpaca_API_connector import api
from Algo_trader_V1.api.alpaca_API_connector import portfolio_overview
from Algo_trader_V1.live_model_functions.AV_get_intraday_stock_no_mtplt import pull_intraday_data, submit_order

stock_symbol = 'AAPL'
def ma_loop(equities_list):

    """
    Parameters
    -----------------
    equities_list : iterable list of strings representing stocks
    interval : 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    time_period : time_period=60, time_period=200
    series_type : close, open, high, low
    datatype : 'json', 'csv', 'pandas'
    """

    while True:
        # iteration start
        for stock_symbol in equities_list:
            '''endless loop for buying and selling; sleep time at the end'''
            # create a iterable tuple for the orders;
            start_time = time.time()
            # return a list with 2 elements (pandas df, dict with info)
            last_price, _ = pull_intraday_data(symbol=stock_symbol,
                                            interval='5min',
                                            outputsize='full',
                                            output_format='pandas')
            # calculate the mean price of the last 25 min of the trading day
            mean_price = last_price['open'][:5].mean()
            # retrieve the very last quote to compare with
            actual_price = last_price['open'][:1].mean()
            # retrieve accounts remaining buying power
            bp = float(api.get_account().buying_power)
            portfolio = portfolio_overview()

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
            sma_50, _ = ti.get_sma(symbol=stock_symbol, interval='daily', time_period='50', series_type='open')
            sma_200, _ = ti.get_sma(symbol=stock_symbol, interval='daily', time_period='200', series_type='open')

            key_list = sorted(enumerate(sma_50.keys()), reverse=False)[:3]
            key_list_2 = sorted(enumerate(sma_200.keys()), reverse=False)[:3]
            # access tuples inside list with key_list[LIST_INDEX][TUPLE_ELEMENT] (both 0-indexed)

            # check if sma_50 is intersecting sma_200 coming from below
            if (sma_50[key_list[2][1]]['SMA'] < sma_200[key_list_2[2][1]]['SMA'] and
                    sma_50[key_list[0][1]]['SMA'] > sma_200[key_list_2[0][1]]['SMA']):
                # buy signal
                print("Executing buy signal...")
                print(stock_symbol, " is being bought")
                try:
                    print("Stock ", stock_symbol, " is being purchased")
                    submit_order(symbol=stock_symbol,
                                 qty=2,
                                 side='buy',
                                 order_type='limit',
                                 time_in_force='gtc',
                                 limit_price=actual_price
                                 )
                    print("script execution time:", time.time() - start_time, " sec.")
                except BaseException as e:
                    print(e)
                    submit_order(symbol=stock_symbol,
                                 qty=float(last_price['high'].head(1) / bp * 0.1),
                                 side='buy',
                                 order_type='limit',
                                 time_in_force='gtc',
                                 limit_price=actual_price
                                 )
            # check if sma_50 is intersecting sma_200 coming from above; the stock is owned; at least one stock is owned
            elif (sma_50[key_list[2][1]]['SMA'] > sma_200[key_list_2[2][1]]['SMA'] and
                    sma_50[key_list[0][1]]['SMA'] < sma_200[key_list_2[0][1]]['SMA']) and\
                    (stock_symbol in portfolio and portfolio[1] > 0):
                # sell signal
                print("Executing sell signal...")
                print("Stock ", stock_symbol, " is being sold")
                try:
                    print("Stock ", stock_symbol, " is being sold")
                    submit_order(symbol=stock_symbol,
                                 qty=2,
                                 side='sell',
                                 order_type='limit',
                                 time_in_force='gtc',
                                 limit_price=mean_price
                                 )
                except BaseException as e:
                    print(e)
                    submit_order(symbol=stock_symbol,
                                 qty=3,
                                 side='sell',
                                 order_type='limit',
                                 time_in_force='gtc',
                                 limit_price=mean_price
                                 )
                    pass
                print("script execution time:", time.time() - start_time, "sec.")
            else:
                print("No action needed to be conducted at", dt.now().isoformat())

            print("Break time of 60s before next stock check to avoid Alpha Vantage overload")
            # break after each iterator element
            time.sleep(61)
        # time in seconds
        time.sleep(17280)

if __name__ == '__main__':
    print("invoked directly; executing script...")
    stock_list_ma = ['AAPL', 'TSLA', 'GOOG', 'NVDA']
    ma_loop(equities_list=stock_list_ma)