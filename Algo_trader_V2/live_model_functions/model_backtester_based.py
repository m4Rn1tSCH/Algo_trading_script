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

from Algo_trader_V2.api.alpaca_py_api import latest_stock_price, submit_market_order, submit_limit_order, get_all_positions


# "C:\Users\Administrator\Documents\file_drop\stock_backtesting_2024-03-06.csv"
def bt_buyer(stocks):
    """
    Data based on backtester module will feed into this function
    and then will trigger buy orders with limit prices.
    Return filter will be applied in starter.py.
    :param stocks: dict; contains all stocks as keys and holding times as values
    :return: None
    """
    if len(stocks.keys()) != 0:
        try:
            tc = TradingClient(api_key=config('ALPACA_API_KEY'), secret_key=config('ALPACA_SECRET_KEY'), paper=True)
            print("buying power on account: ", float(tc.get_account().buying_power))
            # bp = float(tc.get_account().buying_power)
            # set limit price at 98% of theoretically predicted profit
            # TODO: refine logic for selling existing positions
            # li = get_all_positions()
            # acc_pos = [st.symbol for st in li]
            # sell_dict = {}
            # held_pos_list = [s for s in sell_dict.keys() if s in acc_pos]

            # st = stock; ret = return from backtester
            for st, ret in stocks.items():
                # last_price = latest_stock_price(input_list=st)
                # submit_limit_order(symbol=st,
                #                    limit_pr=(last_price * (1 + ret / 100)) * 0.98,
                #                    purchase_notional=0.05 * bp,
                #                    order_side=OrderSide.BUY)
                submit_market_order(symbol=st, quantity=1, order_side=OrderSide.BUY)
                time.sleep(2)
                print(f"backtester buy order executed: {st}", dt.now().isoformat())
        except BaseException as e:
            print(e)
    else:
        print("no stocks in dictionary; no purchases necessary")
    return 'buyer completed task'

if __name__ == '__main__':
    # test successful 9/24/2024
    li = get_all_positions()
    acc_pos = [st.symbol for st in li]
    test_dict = {
        'MSFT': 0.1,
        'NVDA': 0.075,
        'SPY': 0.0885,
        'AAA': 0.254,
        'CAC': 0.065,
        'TGT': 0.0484,
        'VTC': 0.0115,
    }
    filter_dict = {}
    sell_dict = {}
    for k, v in test_dict.items():
        if v >= 0.075:
            filter_dict[k] = v
        else:
            sell_dict[k] = v

    held_pos_list = [s for s in sell_dict.keys() if s in acc_pos]
    bt_buyer(stocks=filter_dict)
