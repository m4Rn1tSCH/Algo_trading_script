# -*- coding: utf-8 -*-
"""
Created on 9/27/2020; 9:23 AM

@author: Bill Jaenke
"""
import time
from datetime import datetime as dt

from alpha_vantage.techindicators import TechIndicators

from Algo_trader_V1.api import alpaca_API_connector as api
from Algo_trader_V1.live_model_functions.AV_get_intraday_stock import pull_intraday_data, submit_order


# loop based on the WEIGHTED MOVING AVERAGE
# this loop does not allow shorting
def wma_loop(equities_list):

    """
    symbol : 'XXXX'
    interval : 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    time_period : time_period=60, time_period=200
    series_type : close, open, high, low
    datatype : 'json', 'csv', 'pandas'
    """

    while True:
        '''
        ACCESS OF WEIGHTED MOVING AVERAGES AND CONSECUTIVE INTERSECTION THEREOF
        naming of day + 1 is inverted to index position because list is in descending order
        then the increasing index element in the list represents a day
        further in the past (smaller date number)
        '''
        # iteration start
        for stock_symbol in equities_list:
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

            # zero indexed counter with values selected before index 3(last element exclusive); start at index 0

            key_list = sorted(enumerate(wma_50.keys()), reverse=False)[:3]
            key_list_2 = sorted(enumerate(wma_200.keys()), reverse=False)[:3]
            # access tuples inside list with key_list[LIST_INDEX][TUPLE_ELEMENT] (both 0-indexed)
            # comparison loop
            if (wma_50[key_list[2][1]]['WMA'] < wma_200[key_list_2[2][1]]['WMA'] and
                    wma_50[key_list[0][1]]['WMA'] > wma_200[key_list_2[0][1]]['WMA']):
                # buy signal
                print("Executing buy signal...")
                print(stock_symbol, " is being bought")
                try:
                    print("Stock is being purchased")
                    submit_order(symbol=stock_symbol,
                                 qty=float(last_price['high'].head(1) / bp * 0.1),
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
            # check if wma_50 is smaller than wma_200; the stock is owned; at least one stock is owned
            elif (wma_50[key_list[2][1]]['WMA'] > wma_200[key_list_2[2][1]]['WMA'] and
                    wma_50[key_list[0][1]]['WMA'] < wma_200[key_list_2[0][1]]['WMA']) and\
                    (stock_symbol in portfolio_list and portfolio_list[1] > 0):
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
                                 qty=float(last_price['high'].head(1) / bp * 0.1),
                                 side='sell',
                                 type='limit',
                                 time_in_force='gtc',
                                 limit_price=actual_price
                                 )
                    pass
            else:
                print("No action conducted at", dt.now().isoformat())
        # time in seconds
        time.sleep(17280)
if __name__ == '__main__':
    print("invoked directly; executing script...")
    portfolio_overview()
    wma_loop(equities_list=stock_list_wma)