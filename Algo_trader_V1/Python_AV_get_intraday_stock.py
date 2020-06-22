"""
This module connects to the integrated Alpha Vantage API and allows data queries
IMPORTANT LIMIT AV API 5 API requests per minute and 500 requests per day
pull_stock_data = pulls data in intervals of days or higher
pull_intraday_data = dataframes for minute intervals

returns a dataframe
"""
from Python_alpaca_API_connector import api


#pull stock data from Alpha Vantage
def pull_stock_data(symbol, adjusted, outputsize, cadence, output_format):
    """
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    adjusted: 'True' or 'False'
    outputsize: 'compact'(100 rows for faster calls) or 'Full'(full retrieval)
    cadence: 'daily' / 'weekly' / 'monthly'
    output_format: ['json', 'csv', 'pandas']
    """
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
                                        "4. close": "close",
                                        "5. adjusted close": "adjusted_close",
                                        "6. volume": "volume",
                                        "7. dividend amount": "dividend amount"},
                                inplace=False)

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
     time_in_force: 'gtc' / 'day',
     limit_price: Any = fl32 (with or without ''),
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
#split up charts into 2 subplots again (prob has been fixed)
import matplotlib.pyplot as plt
import seaborn as sns

# LINE VALUES
#   supported values are: '-', '--', '-.', ':',
#   'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
# Plot the date on x-axis and open price on y-axis

fig, ax = plt.subplots(2, 1, figsize=(15, 8))

ax[0].plot(intra_df.index.values, intra_df['open'], color='green', lw=1, ls='dashdot', marker='solid', label="Open Price")
# Plot the date on x-axis and the trading volume on y-axis
#plt.set_title('Trading Volume', style='oblique')
ax[1].plot(intra_df.index.values, intra_df['volume'], color='orange', lw=1, ls='solid', marker='x', label="Trade Volume")
plt.title('Open Price', style='oblique')
plt.xticks(rotation=60)
plt.show()

