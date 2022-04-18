# -*- coding: utf-8 -*-
"""
Created on 7/12/2021; 12:05 PM

@author: Bill Jaenke
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import normaltest
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from Algo_trader_V2.support_functions.support_features import *
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                proxy=None)

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# this ARIMA script is for investigation purposes of new ideas
# retrieve data
# outputsize=full; all data or outputsize=compact; 100 rows
df, _ = ts.get_daily(symbol='NVDA', outputsize='full')
df = df.rename(columns={"1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. volume": "Volume"
                        }, inplace=False)
df.reset_index(drop=False, inplace=True)
# pandas at 1.2.2 and numpy 1.20.3 at avoids error of conditional import
# TODO: remove redundant columns to reduce VIF, correlation
processed_df = pred_feat(df=df)
# reverse df as it starts with the latest day
processed_df = processed_df.iloc[::-1]
print(processed_df.head())

sns.lineplot(x="date", y="Open", legend='full', data=processed_df)
plt.show()

sns.lineplot(x="date", y="Volume", legend='full', data=processed_df)
plt.show()

sns.boxplot(x="date", y="Open", data=processed_df)
plt.show()

# create training data
train_df = processed_df[:2500]
train_df = train_df.set_index('date')
train_df['Close'] = train_df['Close'].astype(float)

# look for missing values and NaNs that ruin the prediction
print("Number of Nan: ", train_df.isna().sum())
print("Number of nulls: ", train_df.isnull().sum())

# check for seasonality
result = seasonal_decompose(train_df['Open'], model='additive', freq=365)
# decompose plot definitely shows seasonality
# that makes ARIMA necessary
fig = plt.figure(figsize=(15, 12))
fig = result.plot()
# fig.set_size_inches(15, 12)
plt.show()

test_stationarity(train_df['Open'])

# check for viability
# try tutorial and logarithmic approach
df_diff = train_df.Open - train_df.Open.shift(1)
df_diff = df_diff.dropna(inplace=False)
# after shift
test_stationarity(df_diff, window=12)

# index needs to be continuous date range
train_df.index.isnull().sum()

# plot regular df
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_df.Open, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_df.Open, lags=40, ax=ax2)
plt.show()

#  plot diff df
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_diff, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_diff, lags=40, ax=ax2)
plt.show()

# Here we can see the ACF and PACF both has a recurring pattern approximately every 10 periods.
# Indicating a weekly pattern exists.
# Any time you see a regular pattern like that in one of these plots, you should suspect that there is some sort of
# significant seasonal thing going on. Then we should start to consider SARIMA to take seasonality into account

# pick p, d, q
# i/q: order that has made the df stationary (our case: first order)
# AR or p: lag length that is statistically significant with the Dickey-Fuller Test (our case: 10 periods)
# When the AR model is appropriately specified, the residuals from this model can be used
# to directly observe the uncorrelated error
# pass as tuple: (p, 0, q)

# SUMMARY OF NON-STATIONARY DF (FOR COMPARISON)
arima_mod = sm.tsa.ARIMA(train_df.Open, trend='n', order=(5, 1, 0)).fit()
print(arima_mod.summary())

resid = arima_mod.resid
print(normaltest(resid))
# returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
# the residual is not a normal distribution


fig = plt.figure(figsize=(12, 8))
ax0 = fig.add_subplot(111)
sns.distplot(resid, fit=stats.norm, ax=ax0)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.show()

# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(resid)
plt.plot(mu, sigma)
plt.show()

# ACF and PACF
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arima_mod.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arima_mod.resid, lags=40, ax=ax2)

# Here we can see the acf and pacf both has a recurring pattern approximately every 10 periods.
# Indicating a weekly pattern exists.
# Any time you see a regular pattern like that in one of these plots, you should suspect that there is some sort of
# significant seasonal thing going on. Then we should start to consider SARIMA to take seasonality into account

# pick p, d, q
# i/q: order that has made the df stationary (our case: first order)
# AR or p: lag length that is statistically significant with the Dickey-Fuller Test (our case: 10 periods)
# When the AR model is appropriately specified, the residuals from this model can be used
# to directly observe the uncorrelated error
# pass as tuple: (p, 0, q)

# PREDICTION AND EVALUATION
# ARIMA needs all dates to be consecutive and filled with values
# use forward fill .ffill() to fill weekends and holidays
start_index = 0
end_index = len(train_df)
train_df.reset_index(drop=False, inplace=True)
train_df.ffill()
train_df = train_df.set_index('date')

# test for missing values and continuous date index
train_df.isnull().sum().sum()
train_df.index.isnull().sum()

train_df['forecast'] = arima_mod.predict(dynamic=False)
train_df[start_index:end_index][['Open', 'forecast']].plot(figsize=(14, 10))
plt.show()

smape_kun(train_df[start_index:end_index]['Open'], train_df[start_index:end_index]['forecast'])
