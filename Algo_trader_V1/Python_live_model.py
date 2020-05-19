# -*- coding: utf-8 -*-
"""
Created on 18 5/18/2020 8:15 PM 2020

@author: bill-
"""
from datetime import datetime as dt
import pandas as pd
import numpy as np
import time
from Python_AV_get_intraday_stock import pull_intraday_data, pull_stock_data, submit_order

"""
-pull data every hour /every day
-prepare df and decide what to sell or buy
if loop -> buy or sell
append data every hour
calculate/or mark -> sell or buy order
"""
# TODO
# loop works
intra_df = pull_intraday_data(symbol='TSLA',
                              interval='5min',
                              outputsize='full',
                              output_format='pandas')
intra_df['open_diff'] = intra_df['open'].diff()
stock_df = pull_stock_data(symbol='NVDA',
                           adjusted=True,
                           outputsize='full',
                           cadence='monthly',
                           output_format='pandas')
stock_df['open_diff'] = stock_df['open'].diff()

# add feature engineering columns that yield more accuracy

#pred_feat(df=intra_df)
#print(intra_df.head(10))


def trading_support_resistance(df, bin_width=30):


    """
    create empty indicator columns and add values as the data evovles
    sup_tol : supportive tolerance
    res_tol : resistance tolerance
    sup_count : support count
    res_count : resistance count
    sup : support
    res : resistance
    positions : positions open
    signal : signals found
    """

    df['sup_tol'] = pd.Series(np.zeros(len(df)))
    df['res_tol'] = pd.Series(np.zeros(len(df)))
    df['sup_count'] = pd.Series(np.zeros(len(df)))
    df['res_count'] = pd.Series(np.zeros(len(df)))
    df['sup'] = pd.Series(np.zeros(len(df)))
    df['res'] = pd.Series(np.zeros(len(df)))
    df['positions'] = pd.Series(np.zeros(len(df)))
    df['signal'] = pd.Series(np.zeros(len(df)))
    in_support = 0
    in_resistance = 0

    for x in range((bin_width - 1) + bin_width, len(df)):
        df_section = df[x - bin_width:x + 1]

    support_level = min(df_section['open'])
    resistance_level = max(df_section['open'])
    range_level = resistance_level - support_level

    df['res'][x] = resistance_level
    df['sup'][x] = support_level
    #allow a 20% buffer back into the mean zone of the price movement
    df['sup_tol'][x] = support_level + 0.2 * range_level
    # allow a 20% buffer back into the mean zone of the price movement
    df['res_tol'][x] = resistance_level - 0.2 * range_level

    if df['open'][x] >= df['res_tol'][x] and df['open'][x] <= df['res'][x]:
        in_resistance += 1
        df['res_count'][x] = in_resistance
    elif df['open'][x] <= df['sup_tol'][x] and df['open'][x] >= df['sup'][x]:
        in_support += 1
        df['sup_count'][x] = in_support
    else:
        in_support = 0
        in_resistance = 0

    if in_resistance > 2:
        df['signal'][x] = 1
    elif in_support > 2:
        df['signal'][x] = 0
    else:
        df['signal'][x] = df['signal'][x - 1]
        df['positions'] = df['signal'].diff()


    # trading_support_resistance(goog_df_signal)

def test():
    while True:
        print("I love tati")
        # sleep time in minutes
        time.sleep(2)
