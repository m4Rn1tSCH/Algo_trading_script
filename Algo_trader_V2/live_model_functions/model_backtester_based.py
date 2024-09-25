"""
# Created on: 3/29/2024; 11:31 AM

# Created by: bjaen
"""
"""
This model has an approach that uses the backtester data
that will be executed on the EC2 every day at 6.00AM (runtime around 5min)
the best stocks profits will be turned into an order with the corresponding hold time
"""
from datetime import datetime as dt
from decouple import config
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus

from Algo_trader_V2.api.alpaca_py_api import latest_stock_price, submit_market_order, submit_limit_order


# "C:\Users\Administrator\Documents\file_drop\stock_backtesting_2024-03-06.csv"
def bt_buyer(stocks):
    """
    Data based on backtester module will feed into this function
    and then will trigger buy orders with limit prices.
    Return filter will be applied in starter.py.
    :param stocks: dict; contains all stocks as keys and holding times as values
    :return: None
    """

    try:
        tc = TradingClient(api_key=config('ALPACA_API_KEY'), secret_key=config('ALPACA_SECRET_KEY'), paper=True)
        print("buying power on account: ", float(tc.get_account().buying_power))
        bp = float(tc.get_account().buying_power)
        # iterate through list and execute orders
        # execute buy
        # set limit price at 98% of theoretically predicted profit
        # st = tock; ret = return from backtester
        for st, ret in stocks.items():
            last_price = latest_stock_price(input_list=st)
            submit_limit_order(symbol=st,
                               limit_pr=(last_price * (1 + ret / 100)) * 0.98,
                               purchase_notional=0.05 * bp,
                               order_side=OrderSide.BUY)

            # if limit does not work, use market order for now
            # submit_market_order(symbol=st, quantity=1, order_side=OrderSide.BUY)
            # time.sleep(2)
            print(f"backtester order executed: {st}", dt.now().isoformat())
    except BaseException as e:
        print(e)
    return
