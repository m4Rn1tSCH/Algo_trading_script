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
from live_model_functions.Python_live_model import ma_loop

def endpoint_algo():
	ma_loop(stock_symbol='TSLA')
	return 'function run'


print("test was successful")

