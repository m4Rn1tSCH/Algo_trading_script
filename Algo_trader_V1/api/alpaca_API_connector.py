"""
This module connects to the alpaca API and automatically to the Alpha Vantage API
The functions gather data around the account primarily profit n loss; buying power
The keys are stored in  alpaca_acc_config.py
"""
# access files
import alpaca_trade_api as tradeapi
from decouple import config
from Algo_trader_V1.api import alpaca_acc_config

# initialize the API connection

api = tradeapi.REST(config('ALPACA_API_KEY'),
                    config('ALPACA_SECRET_KEY'),
                    'https://paper-api.alpaca.markets')

# Get our account information.
def acc_report():

    account_info = api.get_account()
    print('${} is available as buying power.'.format(account_info.buying_power))
    return account_info

# Lists currently open trades
def list_positions():

    '''
    Print the dictionary generated by the website to list positions
    :return: dictionary
    '''

    try:
        positions = api.list_positions()
        if len(positions) == 0:
            print("no positions found")
        else:
            pass
    except BaseException as e:
        print(e)
    return positions

# account overview with positions as lists
# unlike list
def portfolio_overview():

    '''
    Shows a list of tuples of stock positions owned and corresponding volume in pcs
    :return: list of tuples
    '''

    pos = api.list_positions()
    portfolio_list = []
    print("Current portfolio positions:\n SYMBOL | NO. STOCKS")
    for i in range(0, len(pos), 1):
        # print as tuple
        print((pos[i].symbol, pos[i].qty))
        # append a tuple with the stock and quantity held
        portfolio_list.append((pos[i].symbol, pos[i].qty))
    return portfolio_list

def list_orders():

    try:
        orders = api.list_orders()
        if len(orders) == 0:
            print("no orders found")
        else:
            print("orders are being retrieved")
            pass
    except BaseException as e:
        print(e)
    return orders

# Places a limit order
# this one works; order is visible
# api.submit_order('AAPL', 1, 'buy', 'limit', 'gtc', 170.50)

def get_acc_profit():

    '''
    Check daily equity difference of the account
    :return:
    '''
    # Get account info
    account = api.get_account()

    # Check our current balance vs. our balance at the last market close
    balance_change = float(account.equity) - float(account.last_equity)
    print('Today\'s portfolio balance change: $ {}'.format(balance_change))
    return


if __name__ == '__main__':
    """
    With the Alpaca API, you can check on your daily profit or loss by
    comparing your current balance to yesterday's balance.
    """
    # run two functions when invoked directly
    acc_report()
    get_acc_profit()

