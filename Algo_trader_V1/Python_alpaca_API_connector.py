# access files
import alpaca_trade_api as tradeapi
import acc_config
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import pandas as pd
#initialize the API connection
api = tradeapi.REST(acc_config.API_KEY,
                    acc_config.SECRET_KEY,
                    'https://paper-api.alpaca.markets')



# Get our account information.
def acc_report():

    account_info = api.get_account()
    print(account_info)
    print('${} is available as buying power.'.format(account_info.buying_power))
    return 'Account ready'

# Lists currently open trades
def list_positions():
    api = tradeapi.REST(acc_config.API_KEY,
                        acc_config.SECRET_KEY,
                        'https://paper-api.alpaca.markets')

    try:
        positions = api.list_positions()
        if len(positions) == 0:
            print("no positions found")
        else:
            pass
    except BaseException as e:
        print(e)
    return positions


def list_orders():

    try:
        orders = api.list_orders()
        if len(orders) == 0:
            print("no orders found")
        else:
            pass
    except BaseException as e:
        print(e)
    return orders

# Places a limit order
# this one works; order is visible
#####api.submit_order('AAPL', 1, 'buy', 'limit', 'gtc', 170.50)

def get_acc_profit():
    if __name__ == '__main__':
        """
        With the Alpaca API, you can check on your daily profit or loss by
        comparing your current balance to yesterday's balance.
        """

        # Get account info
        account = api.get_account()

        # Check our current balance vs. our balance at the last market close
        balance_change = float(account.equity) - float(account.last_equity)
        print(f'Today\'s portfolio balance change: ${balance_change}')


# list orders
###api.list_orders()
# Print today's price of a stock
###api.alpha_vantage.historic_quotes(symbol='AAPL')

# ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas')
# data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

# data.to_csv('stock_data.csv')

# Lists all open orders
# orders = api.list_orders()
