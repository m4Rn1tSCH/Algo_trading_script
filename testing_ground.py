"""
Created on: 4/17/2023; 8:28 AM

Created by: Bill J
"""
"""This script is only to test functions syntax or small snippets"""
from Algo_trader_V2.api.alpaca_py_api import *
from Algo_trader_V2.api.alpaca_py_api import market_data

last_price = market_data(symbol_input=['SPY'],
                    time_frame=TimeFrame.Day,
                    start=dt.today() - timedelta(days=365), end=dt.today(),
                    limit=None,
                    sort='desc'
                    )

exec_price = float(last_price['open'].iloc[0]) * 1.025