# -*- coding: utf-8 -*-
"""
Created on 7/23/2020 5:46 PM

@author: bill-
"""
'''
This endpoint starts the algorithm.
Specify which loop and stock
'''
from Algo_trader_V2.live_model_functions.ma import ma_loop

# stocks_picked = ['BILL', 'CUK', 'OIL_CRUDE', 'AAPL', 'NVDA']
if __name__ == '__main__':

	"""
	This functions kicks off the loop in the algorithm and sits on top of everything.
	It just starts the function ma_loop, the moving average loop.
	-----
	Parameters.
	equities_list: strings; specify the stocks for which data will be pulled and
	the check will be performed.
	"""
	# prepare data from stock_skimmer
	# test_dict = zip(df_test_bed['stock'].to_list(), df_test_bed['return'].to_list())
	# test_dict = dict(zip(df_test_bed['stock'].to_list(), df_test_bed['return'].to_list()))
	# top5_l = [k for k, v in test_dict.items() if v > 0]

	ma_loop(equities_list=['MGRX', 'MSFT', 'LEVI', 'AAPL', 'NVDA'])
	print("Algo script is running...")
