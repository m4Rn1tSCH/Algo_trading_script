"""
# Created on: 3/29/2024; 11:31 AM

# Created by: bjaen
"""
"""
This model has an approach that uses the backtester data
that will be executed on the EC2 every day at 6.00AM (runtime around 5min)
the best stocks profits will be turned into an order with the corresponding hold time
"""
from Algo_trader_V2.live_model_functions.AV_get_intraday_stock_no_mtplt import submit_order, pull_intraday_data
from datetime import datetime as dt

# "C:\Users\Administrator\Documents\file_drop\stock_backtesting_2024-03-06.csv"
def bt_buyer(stocks):
    """
    Data based on backtester module will feed into this function
    and then will trigger buy orders with limit prices.
    Return filter will be applied in starter.py.
    :param stocks: dict; contains all stocks as keys and
                       holding times as values
    :return: None
    """
    try:

        # iterate through list and execute orders
        # execute buy
        # set limit price at 98% of theoretically predicted profit
        # st = tock; ret = return from backtester
        for st, ret in stocks.items():
            price = pull_intraday_data(symbol=st, interval='30min',
                               outputsize='compact', output_format='pandas')
            last_price = price['open'][:1]
            submit_order(symbol=st,
                         qty=1,
                         side='buy',
                         type_order='limit',
                         time_in_force='gtc',
                         limit_price=(last_price * (1 + ret / 100)) * 0.98
                         )
            print(f"backtester order executed: {st}", dt.now().isoformat())
    except BaseException as e:
        print(e)
    return
