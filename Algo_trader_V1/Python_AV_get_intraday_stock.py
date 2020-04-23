#OLD WAY IN AV TO PULL DATA

#from pprint import pprint
#import pandas as pd
#ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas')
#data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

#data.to_csv('stock_data.csv')
from Python_alpaca_API_connector import api

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
    except BaseException as e:
        print(e)
        print("API not properly connected")
    return df_pull

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
                                        output_format=output_format)
    except BaseException as e:
        print(e)
        print("API not properly connected")
    return df_intra_pull

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