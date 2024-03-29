# -*- coding: utf-8 -*-
"""
Created on 6/15/2021; 5:07 PM

@author: Bill Jaenke
"""
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from Algo_trader_V2.api.alpaca_API_connector import api
from Algo_trader_V2.support_functions.support_features import pred_feat, trading_support_resistance
from Algo_trader_V2.ml_modules.Python_AI_logic import pipeline_mlp_reg, pipeline_reg, pipeline_rfr, rfe_cross_val


ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                proxy=None)

# outputsize=full; all data or outputsize=compact; 100 rows
df, _ = ts.get_daily_adjusted(symbol='NVDA', outputsize='full')
# rename columns names for better handling
df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low",
                        "4. close": "Close", "5. adjusted close": "Adjusted_close",
                        "6. volume": "Volume", "7. dividend amount": "Dividend_amount",
                        "8. split coefficient": "Split_coefficient"},
               inplace=False)
df = df.drop(columns=['Adjusted_close', 'Dividend_amount', 'Split_coefficient'])
df.reset_index(drop=False, inplace=True)
processed_df = pred_feat(df=df)
print(processed_df.head())

# regression of stock close price
processed_df['Close'].astype('float32')
# MLP
# pipeline_mlp_reg(label_col='Close', df=processed_df, pca_plot=False, shuffle_data=False)
# RFR
# pipeline_rfr(label_col='Close', df=processed_df, pca_plot=False, shuffle_data=False)
# RFE
# rfe_cross_val(label_col='Close', df=processed_df, pca_plot=False, shuffle_data=False)

# apply RNN for stock price prediction

