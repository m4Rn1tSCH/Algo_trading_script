"""
This module connects to the integrated Alpha Vantage API and allows data queries
IMPORTANT LIMIT AV API 5 API requests per minute and 500 requests per day
pull_stock_data = pulls data in intervals of days or higher
pull_intraday_data = dataframes for minute intervals

returns a dataframe
"""
import matplotlib.pyplot as plt

from Algo_trader_V1.api.Python_alpaca_API_connector import api
# TODO
# fix the missing alpha vantage artefacts and replace all lines
# consult AV documentation for new data


# pull stock data from Alpha Vantage
def pull_stock_data(symbol, adjusted, outputsize, cadence, output_format, plot_price=False):
    """
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    adjusted: 'True' or 'False'
    outputsize: 'compact'(100 rows for faster calls) or 'Full'(full retrieval)
    cadence: 'daily' / 'weekly' / 'monthly'
    output_format: ['json', 'csv', 'pandas']
    """
    try:
        df = api.alpha_vantage.historic_quotes(symbol=symbol,
                                                adjusted=adjusted,
                                                outputsize=outputsize,
                                                cadence=cadence,
                                                output_format=output_format)
        # drop the date as index to use it
        df = df.reset_index(drop=False, inplace=False)

        # rename columns names for better handling
        df = df.rename(columns={"1. open": "open",
                                        "2. high": "high",
                                        "3. low": "low",
                                        "4. close": "close",
                                        "5. adjusted close": "adjusted_close",
                                        "6. volume": "volume",
                                        "7. dividend amount": "dividend amount"},
                                inplace=False)
        if plot_price:
            # LINE VALUES
            # supported values are: '-', '--', '-.', ':',
            # 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            fig, ax = plt.subplots(2, 1, figsize=(15, 8))
            ax[0].plot(df['date'], df['open'],
                       color='blue', lw=1, ls='dashdot', marker=',', label="Open Price")
            # Plot the date on x-axis and the trading volume on y-axis
            ax[1].plot(df['date'], df['volume'],
                       color='orange', lw=1, ls='--', marker='x', label="Trade Volume")
            ax[0].set_title('Open Price', style='oblique')
            ax[1].set_title('Trading Volume', style='oblique')
            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper right')
            plt.show()
    except BaseException as e:
        print(e)
        print("API not properly connected!")
    return df

def pull_intraday_data(symbol, interval, outputsize, output_format, plot_price=False):


    """
    DOCUMENTATION
    symbol: pick abbreviation in letter strings 'XXXX'
    interval: ['1min', '5min', '15min', '30min', '60min']
    outputsize: 'Full'
    output_format: ['json', 'csv', 'pandas']
    plot_price: boolean; generate a plot with open price and trading volume
    """

    try:
        df = api.alpha_vantage.intraday_quotes(symbol=symbol,
                                                interval=interval,
                                                outputsize=outputsize,
                                                output_format=output_format)
        #drop the date as index to use it
        df = df.reset_index(drop=False, inplace=False)

        df = df.rename(columns={"1. open": "open",
                                                    "2. high": "high",
                                                    "3. low": "low",
                                                    "4. close": "close",
                                                    "5. volume": "volume"},
                                            inplace=False)
        if plot_price:
            # LINE VALUES
            #   supported values are: '-', '--', '-.', ':',
            #   'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            fig, ax_intra = plt.subplots(2, 1, figsize=(15, 8))
            ax_intra[0].plot(df['date'], df['open'],
                       color='red', lw=1, ls='dashdot', marker=',', label="Open Price")
            # Plot the date on x-axis and the trading volume on y-axis
            ax_intra[1].plot(df['date'], df['volume'],
                       color='cyan', lw=1, ls='--', marker='x', label="Trading Volume")
            ax_intra[0].set_title('Open Price', style='oblique')
            ax_intra[1].set_title('Trading Volume', style='oblique')
            ax_intra[0].legend(loc='upper right')
            ax_intra[1].legend(loc='upper right')

            plt.show()
    except BaseException as e:
        print(e)
        print("API not properly connected!")
    return df


def submit_order(symbol, qty, side, order_type, time_in_force, limit_price):
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
                         type=order_type,
                         time_in_force=time_in_force,
                         limit_price=limit_price
                         )
    except BaseException as f:
        print(f)
    return 'Order submitted'

# seems to list all equities
def get_asset_list(status, asset_class):

    """
    Generate a list of all assets
    status:
    asset_class:
    :return:
    list of asset in possession
    """
    try:
        asset_list = api.list_assets(status=status,
                        asset_class=asset_class
                        )
    except BaseException as f:
        print(f)
    return asset_list


