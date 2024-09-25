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
# from Algo_trader_V2.live_model_functions.ma import ma_loop

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
    filter_d = {k, v for k, v in ret_dict.items() if v > 0.05}
    # solution with
    buy_list = []
    sell_list = []
    pos_ret_df = stock_df[stock_df['return'] >= 0.0275]
    neg_ret_df = stock_df[stock_df['return'] <= 0]
    buy_list = [n for n in pos_ret_df['stock'].to_list()]
    sell_list = [m for m in neg_ret_df['stock'].to_list()]
    bt_buyer(stocks=filter_d)
    print("Algo backtester script is running...")