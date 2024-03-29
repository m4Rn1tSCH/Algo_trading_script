# -*- coding: utf-8 -*-
"""
Created on 7/23/2020 5:46 PM

@author: bill-
"""
'''
This endpoint starts the algorithm.
Specify which loop and stock
'''
import pandas as pd
from datetime import datetime as dt
from Algo_trader_V2.live_model_functions.model_backtester_based import bt_buyer
# from Algo_trader_V2.live_model_functions.ma import ma_loop

if __name__ == '__main__':

    """
    This functions kicks off the loop in the algorithm and sits on top of everything.
    It just starts the function ma_loop, the moving average loop.
    -----
    Parameters.
    equities_list: strings; specify the stocks for which data will be pulled and
    the check will be performed.
    """
    """iterative ma loop section - too data intensive for free plans"""
    # tradable_list = nasdaq_equities_tradable()
    # final_list = []
    # for e in tradable_list:
    #    final_list.append(e.symbol)
    # print("No. of equities tradable on Alpaca: ", len(final_list))

    # stock_df = test_sma_strategy_loop(stock_list=final_list, initial_equity=100000,
                                        # commission=0.0015, intraday_data=False, daily_data=True,
                                        # ma_lag1=50, ma_lag2=200)
    # pos_ret_df = stock_df[stock_df['return'] >= 0.0275]
    # prepare data from stock_skimmer
    # test_dict = zip(df_test_bed['stock'].to_list(), df_test_bed['return'].to_list())
    # ret_dictionary = dict(zip(pos_ret_df['stock'].to_list(), pos_ret_df['return'].to_list()))
    # return_list = [k for k, v in ret_dictionary.items() if v > 0.0275]

    # config for task scheduler; will use separate process to generate CSV
    """ma loop section"""
    # today = dt.today().date().isoformat()
    # stock_input_df = pd.read_csv(f'C:/Users/Administrator/Documents/file_drop/stock_backtesting_{today}.csv')
    # eq_list = stock_input_df['stock'].to_list()[:6]
    # ma_loop(equities_list=eq_list)
    # print("Algo ma-loop script is running...")

    """backtester loop section"""
    today = dt.today().date().isoformat()
    st_in_df = pd.read_csv(f'C:/Users/Administrator/Documents/file_drop/stock_backtesting_{today}.csv')
    ret_dict = dict(zip(st_in_df['stock'].to_list(), st_in_df['return'].to_list()))
    filter_d = {k, v for k, v in ret_dict.items() if v > 0.05}
    bt_buyer(stocks=filter_d)
    print("Algo backtester script is running...")