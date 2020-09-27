# -*- coding: utf-8 -*-
"""
Created on 9/27/2020 2020; 9:24 AM

@author: Bill Jaenke
"""
import time

from Algo_trader_V1.live_model_functions.AV_get_intraday_stock import pull_intraday_data, submit_order


# Quick visual confirmation area for picked stocks
# intra_df = pull_intraday_data(symbol='TSLA',
#                               interval='5min',
#                               outputsize='full',
#                               output_format='pandas',
#                               plot_price=False)
# intra_df['open_diff'] = intra_df['open'].diff()

# monthly data
# stock_df = pull_data_adj(symbol='NVDA',
#                        outputsize='full',
#                        cadence='monthly',
#                        output_format='pandas',
#                        plot_price=False)
# stock_df['open_diff'] = stock_df['open'].diff()

# time loop for trading logic


def simple_loop():

    """
    Thank you for using Alpha Vantage!
    Our standard API call frequency is 5 calls per minute and 500 calls per day.
    Premium: https://www.alphavantage.co/premium/
    """
    while True:
        last_price = pull_intraday_data(symbol='TSLA',
                                        interval='5min',
                                        outputsize='full',
                                        output_format='pandas')
        # calculate the mean price of the last 25 min of the trading day
        mean_price = last_price['open'][:5].mean()
        # retrieve the very last quote to compare with
        actual_price = last_price['open'][:1].mean()
        print("Price retrieved")
        if mean_price > actual_price:
            # buy signal
            try:
                print("Stock is being purchased")
                submit_order(symbol='TSLA',
                             qty=2,
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol='TSLA',
                             qty=2,
                             side='buy',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=mean_price
                             )
        elif mean_price < actual_price:
            try:
                print("Stock is being sold")
                submit_order(symbol='TSLA',
                             qty=2,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
            except BaseException as e:
                print(e)
                submit_order(symbol='TSLA',
                             qty=2,
                             side='sell',
                             type='limit',
                             time_in_force='gtc',
                             limit_price=actual_price
                             )
        else:
            print("Both prices identical; no action")
        # loop will pause for x seconds
        time.sleep(17000)
