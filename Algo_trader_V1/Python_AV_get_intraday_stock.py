#OLD WAY IN AV TO PULL DATA

#from pprint import pprint
#import pandas as pd
#ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas')
#data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

#data.to_csv('stock_data.csv')
from Python_alpaca_API_connector import api

#IMPORTANT  LIMIT AV API 5 API requests per minute and 500 requests per day
#pull stock data from Alpha Vantage
def pull_stock_data(symbol, adjusted, outputsize, cadence, output_format):
    '''
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    adjusted: 'True' or 'False'
    outputsize: 'Full'
    cadence: 'daily' / 'weekly' / 'monthly'
    output_format: ['json', 'csv', 'pandas']
    '''
    try:
        df_pull = api.alpha_vantage.historic_quotes(symbol=symbol,
                                adjusted=adjusted,
                                outputsize=outputsize,
                                cadence=cadence,
                                output_format=output_format)
        #drop the date as index to use it
        df_pull = df_pull.reset_index(drop=False, inplace=False)

        #rename columns names for better handling
        df_pull = df_pull.rename(columns={"1. open": "open",
                                        "2. high": "high",
                                        "3. low": "low",
                                        "4. close":"close",
                                        "5. volume": "volume"},
                                inplace=False)

    except BaseException as e:
        print(e)
        print("API not properly connected")
    return df_pull

# pull_stock_data(symbol='COTY',
#                 adjusted=True,
#                 outputsize='full',
#                 cadence='monthly',
#                 output_format='pandas')

def pull_intraday_data(symbol, interval, outputsize, output_format):
    '''
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    interval: ['1min', '5min', '15min', '30min', '60min']
    outputsize: 'Full'
    output_format: ['json', 'csv', 'pandas']
    '''
    try:
        df_intra_pull = api.alpha_vantage.intraday_quotes(symbol=symbol,
                                        interval=interval,
                                        outputsize=outputsize,
                                        output_format=output_format
                                        )
        #drop the date as index to use it
        df_intra_pull = df_intra_pull.reset_index(drop=False, inplace=False)

        df_intra_pull = df_intra_pull.rename(columns={"1. open": "open",
                                                    "2. high": "high",
                                                    "3. low": "low",
                                                    "4. close": "close",
                                                    "5. volume": "volume"},
                                            inplace=False)

    except BaseException as e:
        print(e)
        print("API not properly connected")
    return df_intra_pull

#intra_df = pull_intraday_data(symbol='COTY',
#                            interval='5min',
#                            outputsize='full',
#                            output_format='pandas')

def submit_order(symbol, qty, side, type, time_in_force, limit_price):
    '''
     DOCUMENTATION
     symbol: Abbr in 'XXX',
     qty: int,
     side: 'buy' / 'sell',
     type: 'limit',
     time_in_force: 'gtc',
     limit_price: Any = fl32,
     stop_price: LIMIT ORDERS DO NOT REQUIRE A STOP PRICE
     '''
    try:
        api.submit_order(symbol=symbol,
                         qty=qty,
                         side=side,
                         type=type,
                         time_in_force=time_in_force,
                         limit_price=limit_price
                         )
    except BaseException as f:
        print(f)
    return 'Order submitted'

# seems to list all equities
def get_asset_list(status, asset_class):
    '''
    Generate a list of all assets
    status:
    asset_class:
    :return:
    list of asset in possession
    '''
    try:
        asset_list = api.list_assets(status=status,
                        asset_class=asset_class
                        )
    except BaseException as f:
        print(f)
    return asset_list
#%%
#TODO
#fix date columns
#split up carts into 2 subplots again (prob has been fixed)
import matplotlib.pyplot as plt
'''
# LINE VALUES
#   supported values are: '-', '--', '-.', ':',
#   'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
# Plot the date on x-axis and open price on y-axis

plt.title('Open Price', style='oblique')
plt.xticks(rotation=60)
intra_df['date'], intra_df['1. open'].plot(color='green', lw=1, ls='dashdot', marker='x', label="Open Price")
# Plot the date on x-axis and the trading volume on y-axis
#plt.set_title('Trading Volume', style='oblique')
intra_df['date'], intra_df['5. volume'].plot(color='orange', lw=1, ls='solid', marker='x', label="Trade Volume")
plt.show()
'''