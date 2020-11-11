# -*- coding: utf-8 -*-
"""
Created on 9/3/2020 2020; 3:22 PM

@author: Bill Jaenke
"""
# just trying out some snippets
from Algo_trader_V1.api.alpaca_API_connector import api
# stocks = ['BILL', 'CUK', 'OIL_CRUDE', 'AAPL', 'NVDA']
# while True:
#
#     for i in stocks:
#         print(i)
#     time.sleep(5)

stocks = ['BILL', 'CUK', 'AAPL', 'NVDA']
# outer infinite loop with inner check
while True:
    while api.get_clock().timestamp > api.get_clock().next_open:
        # if api.get_clock().timestamp < api.get_clock().next_close and api.get_clock().timestamp > api.get_clock().next_open:
        print("open")
    print("closed")
