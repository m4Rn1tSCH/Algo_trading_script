#access files
import alpaca_trade_api as tradeapi
import acc_config
from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import pandas as pd

api = tradeapi.REST('PKONO5YVIY3IDJLD8DZ0',
                    'EOM4O0b/JthgUu5RsmrcqqYVixMHGGM4OpNKRPs6',
                    'https://paper-api.alpaca.markets')

# Get our account information.
account = api.get_account()

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))

# Lists currently open trades
positions = api.list_positions()

# Places a limit order
api.submit_order('AAPL',1,'buy','limit','gtc',170.50)

#Print today's price of a stock
api.alpha_vantage.historic_quotes(symbol='AAPL')

#ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas')
#data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

data.to_csv('stock_data.csv')

# Lists all open orders
orders = api.list_orders()

