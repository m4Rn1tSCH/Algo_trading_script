"""
Created on: 4/17/2023; 8:28 AM

Created by: Bill J
"""
"""This script is only to test functions syntax or small snippets"""
from Algo_trader_V2.api.alpaca_API_connector import *
from Algo_trader_V2.live_model_functions.AV_get_intraday_stock_no_mtplt import pull_intraday_data, submit_order

last_price = pull_intraday_data(symbol='NVDA',
								interval='5min',
								outputsize='full',
								output_format='pandas')[:5]

exec_price = float(last_price['open'].iloc[0]) * 1.025