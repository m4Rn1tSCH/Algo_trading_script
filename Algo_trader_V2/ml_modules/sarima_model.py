# -*- coding: utf-8 -*-
"""
Created on 4/15/2022; 9:24 AM

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

#####
# retrieve data
# outputsize=full; all data or outputsize=compact; 100 rows
df, _ = ts.get_daily(symbol='NVDA', outputsize='full')
# rename columns names for better handling
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
# dropping the index later as well
# processed_df.reset_index(drop=True, inplace=True)
print(processed_df.head())
#####

# processed_df['date'] = pd.to_datetime(processed_df['date'], format="%Y-%m-%d")

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
print(train_df.isna().sum())
print(train_df.isnull().sum())

result = seasonal_decompose(train_df['Open'], model='additive', freq=365)

# decompose plot definitely shows seasonality
# that makes ARIMA necessary
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(15, 12)
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

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_df.Open, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_df.Open, lags=40, ax=ax2)
plt.show()

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

# CONSIDER SEASONALITY BY SARIMA
# Things to note here:
# When running such a large batch of models, particularly when the autoregressive and moving average
# orders become large, there is the possibility of poor maximum likelihood convergence.
# We use the option enforce_invertibility=False, which allows the moving average polynomial to be non-invertible,
# so that more of the models are estimable.
# If several models do not produce good results, their AIC value is set to NaN.
# (Durbin and Koopman note numerical problems with the high order model)

# procedure; pick 10 orders (AR/p value) and see when the first value is statistically significant
# check plot and MAPE/SMAPE for minimal value
sarima_mod = sm.tsa.statespace.SARIMAX(train_df.Open, trend='n', order=(4, 1, 1), enforce_stationarity=True).fit()
print(sarima_mod.summary())

resid = sarima_mod.resid
print(normaltest(resid))

fig = plt.figure(figsize=(12, 8))
ax0 = fig.add_subplot(111)
sns.distplot(resid, fit=stats.norm, ax=ax0)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.show()
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(resid)

# ACF and PACF
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sarima_mod.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sarima_mod.resid, lags=40, ax=ax2)
fig.show()

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
# SARIMA needs all dates to be consecutive and filled with values
# use forward fill .ffill() to fill weekends and holidays
start_index = 0
end_index = len(train_df)
train_df.reset_index(drop=False, inplace=True)
train_df.ffill()
train_df = train_df.set_index('date')

# test for missing values and continuous date index
train_df.isnull().sum().sum()
train_df.index.isnull().sum()
# set dynamic=false to avoid nans
train_df['forecast'] = sarima_mod.predict(dynamic=False)
train_df[start_index:end_index][['Open', 'forecast']].plot(figsize=(14, 10))
plt.show()

smape_kun(train_df[start_index:end_index]['Open'], train_df[start_index:end_index]['forecast'])

# SARIMAX - adding external variables
# start_index = 0
# end_index = len(train_df)
#
# train_df.head()
#
# holiday = pd.read_csv('../input/holiday/USholidays.csv', header=None, names = ['date', 'holiday'])
# holiday['date'] = pd.to_datetime(holiday['date'], yearfirst = True, format = '%y/%m/%d')
# holiday.head()
#
# train_df = train_df.merge(holiday, how='left', on='date')
# train_df['holiday_bool'] = pd.notnull(train_df['holiday']).astype(int)
# train_df = pd.get_dummies(train_df, columns = ['month','holiday','weekday'] ,
#                           prefix = ['month','holiday','weekday'])
# # train_df.head()
# # train_df.shape
# # train_df.columns
#
# ext_var_list = ['date','year', 'day', 'holiday_bool',
#        'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
#        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
#        'holiday_Christmas Day', 'holiday_Columbus Day',
#        'holiday_Independence Day', 'holiday_Labor Day',
#        'holiday_Martin Luther King Jr. Day', 'holiday_Memorial Day',
#        'holiday_New Year Day', 'holiday_Presidents Day (Washingtons Birthday)',
#        'holiday_Thanksgiving Day', 'holiday_Veterans Day', 'weekday_0',
#        'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
#        'weekday_6']
#
# exog_data = train_df[ext_var_list]
# exog_data = exog_data.set_index('date')
# exog_data.head()
#
# train_df = train_df.set_index('date')
# # train_df = train_df.reset_index()
# train_df.head()
#
# start_index = '2017-10-01'
# end_index = '2017-12-31'
# # exog_data.head()
#
# %%time
# sarimax_mod6 = sm.tsa.statespace.SARIMAX(endog = train_df.sales[:start_index],
#                                         exog = exog_data[:start_index],
#                                         trend='n', order=(6,1,0), seasonal_order=(0,1,1,7)).fit()
# print(sarimax_mod6.summary())
#
# start_index = '2017-10-01'
# end_index = '2017-12-30'
# end_index1 = '2017-12-31'
#
# sarimax_mod6.forecast(steps = 121,exog = exog_data[start_index:end_index])
#
# train_df['forecast'] = sarimax_mod6.predict(start = pd.to_datetime(start_index), end= pd.to_datetime(end_index1),
#                                             exog = exog_data[start_index:end_index],
#                                             dynamic= True)
#
# train_df[start_index:end_index][['sales', 'forecast']].plot(figsize=(12, 8))
#
# smape_kun(train_df[start_index:end_index]['sales'],train_df[start_index:end_index]['forecast'])