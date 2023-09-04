# -*- coding: utf-8 -*-
"""
Created on 9/4/2023; 10:22 AM

@author: Bill Jaenke
"""
import time
from datetime import datetime as dt
from datetime import timedelta

from alpha_vantage.techindicators import TechIndicators

from Algo_trader_V2.api.alpaca_API_connector import api
from Algo_trader_V2.api.alpaca_API_connector import portfolio_overview
from Algo_trader_V2.live_model_functions.AV_get_intraday_stock_no_mtplt import pull_intraday_data, submit_order
'''
test iteration to test core conditions for bugs
'''
stock_symbol = ['MSFT']

start_time = time.time()
print("Checking element: ", stock_symbol)
# pull_intraday can return a list with 2 elements (pandas df, dict with info)
last_price = pull_intraday_data(symbol=stock_symbol,
								interval='5min',
								outputsize='full',
								output_format='pandas')[:5]
# retrieve the very last quote; add 2.5% for order execution
exec_price = float(last_price['open'].iloc[0]) * 1.025
# retrieve accounts remaining buying power
bp = float(api.get_account().buying_power)
portfolio = portfolio_overview()
# dynamic quantity scaled by buying power for buy orders
dyn_qty = round(float(bp * 0.1 / last_price['high'].iloc[0]), ndigits=0)
'''
ACCESS OF WEIGHTED MOVING AVERAGES AND CONSECUTIVE INTERSECTION THEREOF
naming of day + 1 is inverted to index position because list is in descending order
then the increasing index element in the list represents a day
further in the past (smaller date number)
'''
# zero indexed counter with values selected before index 3(last element exclusive); start at index 0
# tech indicator returns a tuple; sma dictionary with values; meta dict with characteristics
# instantiate the class first and provide the API key
ti = TechIndicators('PKS7JXWMMDQQXQNDWT2P')
sma_50, _ = ti.get_sma(symbol=stock_symbol, interval='daily', time_period='50', series_type='open')
sma_200, _ = ti.get_sma(symbol=stock_symbol, interval='daily', time_period='200', series_type='open')
# access tuples inside list with key_list[LIST_INDEX][TUPLE_ELEMENT] (both 0-indexed)
key_list = sorted(enumerate(sma_50.keys()), reverse=False)[:3]
key_list_2 = sorted(enumerate(sma_200.keys()), reverse=False)[:3]

# check if sma_50 is intersecting sma_200 coming from below
if (sma_50[key_list[2][1]]['SMA'] < sma_200[key_list_2[2][1]]['SMA']) and\
	(sma_50[key_list[0][1]]['SMA'] >= sma_200[key_list_2[0][1]]['SMA']):
	# buy signal
	try:
		print("Stock buy: ", stock_symbol, "\ntimestamp: ", dt.now())
		submit_order(symbol=stock_symbol,
					 qty=dyn_qty,
					 side='buy',
					 type_order='market',
					 time_in_force='day',
					 limit_price=exec_price)
	except BaseException as e:
		print(e)
		submit_order(symbol=stock_symbol,
					 qty=5,
					 side='buy',
					 type_order='market',
					 time_in_force='day',
					 limit_price=exec_price)
	print("Order successful; script execution time:", time.time() - start_time, " sec.")
# check if sma_50 is intersecting sma_200 coming from above; the stock is owned;
# at least one stock is owned
elif (sma_50[key_list[2][1]]['SMA'] > sma_200[key_list_2[2][1]]['SMA']) and\
	(sma_50[key_list[0][1]]['SMA'] <= sma_200[key_list_2[0][1]]['SMA']) and\
	(stock_symbol in portfolio and portfolio[1] > 0):
	# sell signal
	sell_qty = round(float(bp * 0.05 / last_price['close'].iloc[0]), ndigits=0)
	try:
		print("Stock being sold: ", stock_symbol, "\n timestamp: ", dt.now())
		submit_order(symbol=stock_symbol,
					 qty=sell_qty,
					 side='sell',
					 type_order='market',
					 time_in_force='day',
					 limit_price=exec_price)
	except BaseException as e:
		print(e)
		submit_order(symbol=stock_symbol,
					 qty=sell_qty,
					 side='sell',
					 type_order='market',
					 time_in_force='day',
					 limit_price=exec_price)
		pass
	print("Order successful; script execution time:", time.time() - start_time, " sec")
else:
	print("No action needed to be conducted at: ", dt.now().isoformat())

time.sleep(3)

# this is for direct testing
if __name__ == '__main__':

	print("TEST RUN;\ninvoked directly; executing script...")
	stock_list_ma = ['AAPL', 'TSLA', 'GOOG', 'NVDA']
	# ma_loop(equities_list=stock_list_ma)
