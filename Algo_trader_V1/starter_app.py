# -*- coding: utf-8 -*-
"""
Created on 23 7/23/2020 5:46 PM

@author: bill-
"""
'''
This endpoint unifies all functions
and starts the algorithm.
Specify which loop and stock
'''
from Algo_trader_V1.live_model_functions.Python_live_model import ma_loop, wma_loop


def endpoint_algo():

	"""
	This functions kicks off the loop in the algorithm and should sit on top of everything.
	It just starts the function ma_loop, the moving average loop.
	-----
	Parameters.
	stock_symbol: string; specify the stock for which data will be pulled.
	"""
	ma_loop(stock_symbol='TSLA')
	return 'function run'


print("test was successful")
