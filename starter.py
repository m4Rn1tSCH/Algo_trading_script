# -*- coding: utf-8 -*-
"""
Created on 7/23/2020 5:46 PM

@author: bill-
"""
'''
This endpoint starts the algorithm.
Specify which loop and stocks
'''
import pandas as pd
from datetime import datetime as dt
from Algo_trader_V2.live_model_functions.model_backtester_based import bt_buyer
from Algo_trader_V2.api.alpaca_py_api import get_all_positions

if __name__ == '__main__':

    """
    This functions kicks off the buyer function; the backtesting is done by the backtester module.
    -----
    Parameters.
    stocks: list of strings; specify the stocks for which data will be pulled and
    the check will be performed.
    """
    # config for task scheduler; will use separate process to generate CSV
    """backtester loop section"""
    today = dt.today().date().isoformat()
    st_in_df = pd.read_csv(f'C:/Users/Administrator/Documents/file_drop/stock_backtesting_{today}.csv')
    # solution with dictionary
    ret_dict = dict(zip(st_in_df['stock'].to_list(), st_in_df['return'].to_list()))

    li = get_all_positions()
    acc_pos = [st.symbol for st in li]

    buy_d = {}
    sell_d = {}
    for k, v in ret_dict.items():
        if v > 0.075:
            buy_d[k] = v
        else:
            sell_d[k] = v
    # solution with lists

    held_pos_list = [s for s in sell_d.keys() if s in acc_pos]
    # buyer script
    bt_buyer(stocks=buy_d)
    print("Algo backtester script is running...")
