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
from Algo_trader_V1.live_model_functions.live_model import ma_loop

# stocks_picked = ['BILL', 'CUK', 'OIL_CRUDE', 'AAPL', 'NVDA']
def endpoint_algo():

	"""
	This functions kicks off the loop in the algorithm and sits on top of everything.
	It just starts the function ma_loop, the moving average loop.
	-----
	Parameters.
	equities_list: strings; specify the stocks for which data will be pulled and
	the check will be performed.
	"""

	ma_loop(equities_List=['BILL', 'CUK', 'OIL_CRUDE', 'AAPL', 'NVDA'])
	return 'function run'


print("algo script is running...")
