# -*- coding: utf-8 -*-
"""
Created on 8/11/2021; 9:42 PM

@author: Bill Jaenke
"""
from Algo_trader_V2.support_functions.support_features import pred_feat, trading_support_resistance
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                proxy=None)

# outputsize=full; all data or outputsize=compact; 100 rows
df, _ = ts.get_daily_adjusted(symbol='NVDA', outputsize='full')
# rename columns names for better handling
df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low",
                        "4. close": "Close", "5. volume": "Volume"},
               inplace=False)
df.reset_index(drop=False, inplace=True)
# pandas at 1.2.2 and numpy 1.20.3 at avoids error of conditional import
processed_df = pred_feat(df=df)
# reverse df as it starts with the latest day
processed_df = processed_df.iloc[::-1]
# dropping the index later as well
# processed_df.reset_index(drop=True, inplace=True)
print(processed_df.head())

train_df = processed_df[:2500]
train_df = train_df.set_index('date')
train_df['Close'] = train_df['Close'].astype(float)

# Iterate over all ARIMA(p, q) models with p, q in [0, 6]
aic_full = pd.DataFrame(np.zeros((6, 6), dtype=float))
aic_miss = pd.DataFrame(np.zeros((6, 6), dtype=float))
for p in range(0, 6, 1):
    for q in range(0, 6, 1):
        if p == 0 and q == 0:
            continue
        # Estimate the model with no missing data points
        mod = sm.tsa.statespace.SARIMAX(train_df, order=(p, 0, q), enforce_invertibility=False)
        try:
            res = mod.fit(disp=False)
            aic_full.iloc[p, q] = res.aic
        except:
            aic_full.iloc[p, q] = np.nan

        # Estimate the model with missing data points
        mod = sm.tsa.statespace.SARIMAX(dta_miss, order=(p, 0, q), enforce_invertibility=False)
        try:
            res = mod.fit(disp=False)
            aic_miss.iloc[p, q] = res.aic
        except:
            aic_miss.iloc[p, q] = np.nan


mod = sm.tsa.statespace.SARIMAX(dta_miss, order=(1, 0, 1))
res = mod.fit(disp=False)
print(res.summary())
