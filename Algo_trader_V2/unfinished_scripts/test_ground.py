# -*- coding: utf-8 -*-
"""
Created on 9/3/2020 2020; 3:22 PM

@author: Bill Jaenke
"""
import time

# just trying out some snippets
from Algo_trader_V2.api.alpaca_API_connector import api

# stocks = ['BILL', 'CUK', 'AAPL', 'NVDA']
# while True:
#     for i in stocks:
#         print(i)
#     time.sleep(5)

stocks = ['BILL', 'CUK', 'AAPL', 'NVDA']
# outer infinite loop with inner check
while True:
    while api.get_clock().timestamp < api.get_clock().next_close and \
            api.get_clock().timestamp > api.get_clock().next_open:
        print("open")
    print("Markets closed at: ", api.get_clock().next_close, "Algo is inactive for next: ",\
          (api.get_clock().next_open - api.get_clock().timestamp).seconds, "s")
    time.sleep((api.get_clock().next_open - api.get_clock().timestamp).seconds)
#%%
import alpaca_trade_api
import pandas as pd

alpaca_api = alpaca_trade_api.REST(key_id=key_id, secret_key=secret_key, base_url=endpoint, api_version=None, oauth=None)
start = pd.Timestamp(year=2016, month=1, day=1).isoformat()
start = pd.Timestamp(year=2016, month=1, day=1, tz='US/Central').isoformat() # wrong result with or without timezone

print('getting barset starting from', start)

wrong_bars = alpaca_api.get_barset('AAPL', '5Min', limit=100, start=start)
correct_bars = alpaca_api.get_barset('AAPL', '5Min', start=start)

print('wrong_bars with', wrong_bars['AAPL'][0].t)
#  The current date is printed
#%%
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from Algo_trader_V2.api.alpaca_API_connector import api
from Algo_trader_V2.support_functions.support_features import pred_feat, trading_support_resistance
# from Algo_trader_V2.live_model_functions.AV_get_intraday_stock_no_mtplt import av_intraday, av_daily_adj,\
#     pull_intraday_data, pull_data_adj
ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                proxy=None)
df, _ = ts.get_daily_adjusted(symbol='NVDA', outputsize='compact')
# rename columns names for better handling
df = df.rename(columns={"1. open": "Open",
                            "2. high": "High",
                            "3. low": "Low",
                            "4. close": "Close",
                            "5. adjusted close": "Adjusted_close",
                            "6. volume": "Volume",
                            "7. dividend amount": "Dividend_amount",
                            "8. split coefficient": "Split_coefficient"},
                   inplace=False)
df = df.drop(columns=['Adjusted_close', 'Dividend_amount', 'Split_coefficient'])
df.reset_index(drop=False, inplace=True)
processed_df = pred_feat(df=df)
print(processed_df.head())
trading_support_resistance(df=processed_df, bin_width=30)