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

from Algo_trader_V2.support_functions.support_features import pred_feat, trading_support_resistance
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='IH4EENERLUFUKJRW', output_format='pandas', treat_info_as_error=True, indexing_type='date',
                proxy=None)

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

#####
# retrieve data
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
# pandas at 1.2.2 and numpy 1.20.3 at avoids error of conditional import
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


train_df = processed_df[:2500]
train_df = train_df.set_index('date')
train_df['Close'] = train_df['Close'].astype(float)

# look for missing values and NaNs that ruin the prediction
train_df.isna().sum()
train_df.isnull().sum()
train_df.head()


result = seasonal_decompose(train_df['Open'], model='additive', freq=365)

# decompose plot definitely shows seasonality
# that makes ARIMA necessary
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(15, 12)
plt.show()


def test_stationarity(timeseries, window=12, cutoff=0.01):
    # Determing rolling statistics
    rol_mean = timeseries.rolling(window).mean()
    rol_std = timeseries.rolling(window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rol_mean, color='red', label='Rolling Mean')
    std = plt.plot(rol_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform augmented Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=20)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    print(dfoutput)

test_stationarity(train_df['Open'])

# check for viability
# try tutorial and logarithmic approach
df_diff = train_df.Open - train_df.Open.shift(1)
df_diff = df_diff.dropna(inplace = False)
test_stationarity(df_diff, window = 12)

# index needs to be continuous date range
train_df.index.isnull().sum()

# external test; automatic minimization of auto regression
# from pmdarima import auto_arima
# stepwise_fit = auto_arima(df['AvgTemp'], trace=True,
# suppress_warnings=True)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_df.Open, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_df.Open, lags=40, ax=ax2)
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_diff, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_diff, lags=40, ax=ax2)
plt.show()

# Here we can see the acf and pacf both has a recurring pattern approximately every 10 periods.
# Indicating a weekly pattern exists.
# Any time you see a regular pattern like that in one of these plots, you should suspect that there is some sort of
# significant seasonal thing going on. Then we should start to consider SARIMA to take seasonality into account

# pick p, d, q
# i: order that has made the df stationary (our case: first order)
# AR or p: lag length that is statistically significant with the Dickey-Fuller Test (our case: 10 periods)
# When the AR model is appropriately specified, the the residuals from this model can be used
# to directly observe the uncorrelated error
# pass as tuple: (AR, i, )

# SUMMARY OF NON-STATIONARY DF (FOR COMPARISON)
arima_mod = sm.tsa.ARIMA(train_df.Open, (5,1,0)).fit(disp=False)
print(arima_mod.summary())

resid = arima_mod.resid
print(normaltest(resid))
# returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
# the residual is not a normal distribution


fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)
sns.distplot(resid, fit=stats.norm, ax=ax0)
plt.show()

# TODO - still empty
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(resid)
#Now plot the distribution using
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')
plt.show()

# analyze results
# ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arima_mod.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arima_mod.resid, lags=40, ax=ax2)

# CONSIDER SEASONALITY BY SARIMA
sarima_mod = sm.tsa.statespace.SARIMAX(train_df.Open, trend='n', order=(10, 1, 0),
                                       enforce_stationarity=True).fit()
print(sarima_mod.summary())

resid = sarima_mod.resid
print(normaltest(resid))

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)

sns.distplot(resid ,fit=stats.norm, ax=ax0)
plt.show()
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(resid)

#Now plot the distribution using
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')


# ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sarima_mod.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sarima_mod.resid, lags=40, ax=ax2)
fig.show()

# TODO; all forecasted values become NaN - investigate
# PREDICTION AND EVALUATION
# sarima needs all dates to be consecutive and filled with values
# use forward fill . ffill() to fill weekends and holidays
start_index = 0
end_index = len(train_df)
train_df.reset_index(drop=False, inplace=True)
train_df.ffill()
train_df = train_df.set_index('date')

# test for missing values and continuous date index
train_df.isnull().sum().sum()
train_df.index.isnull().sum()
# TODO - removing start and end renders it 0 instead NAN
train_df['forecast'] = sarima_mod.predict(dynamic=False)
train_df[start_index:end_index][['Open', 'forecast']].plot(figsize=(14, 10))
plt.show()


def smape_kun(y_true, y_pred):
    mape = np.mean(abs((y_true-y_pred)/y_true)) * 100
    smape = np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f'% (mape,smape), "%")

smape_kun(train_df[start_index:end_index]['Open'], train_df[start_index:end_index]['forecast'])

# SARIMAX - adding external variables

start_index = 0
end_index = len(train_df)

train_df.head()

holiday = pd.read_csv('../input/holiday/USholidays.csv',header=None, names = ['date', 'holiday'])
holiday['date'] = pd.to_datetime(holiday['date'], yearfirst = True, format = '%y/%m/%d')
holiday.head()

train_df = train_df.merge(holiday, how='left', on='date')
train_df['holiday_bool'] = pd.notnull(train_df['holiday']).astype(int)
train_df = pd.get_dummies(train_df, columns = ['month','holiday','weekday'] , prefix = ['month','holiday','weekday'])
# train_df.head()
# train_df.shape
# train_df.columns

ext_var_list = ['date','year', 'day', 'holiday_bool',
       'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
       'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
       'holiday_Christmas Day', 'holiday_Columbus Day',
       'holiday_Independence Day', 'holiday_Labor Day',
       'holiday_Martin Luther King Jr. Day', 'holiday_Memorial Day',
       'holiday_New Year Day', 'holiday_Presidents Day (Washingtons Birthday)',
       'holiday_Thanksgiving Day', 'holiday_Veterans Day', 'weekday_0',
       'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',
       'weekday_6']

exog_data = train_df[ext_var_list]
exog_data = exog_data.set_index('date')
exog_data.head()

train_df = train_df.set_index('date')
# train_df = train_df.reset_index()
train_df.head()

start_index = '2017-10-01'
end_index = '2017-12-31'
# exog_data.head()

%%time
sarimax_mod6 = sm.tsa.statespace.SARIMAX(endog = train_df.sales[:start_index],
                                        exog = exog_data[:start_index],
                                        trend='n', order=(6,1,0), seasonal_order=(0,1,1,7)).fit()
print(sarimax_mod6.summary())

start_index = '2017-10-01'
end_index = '2017-12-30'
end_index1 = '2017-12-31'

sarimax_mod6.forecast(steps = 121,exog = exog_data[start_index:end_index])

train_df['forecast'] = sarimax_mod6.predict(start = pd.to_datetime(start_index), end= pd.to_datetime(end_index1),
                                            exog = exog_data[start_index:end_index],
                                            dynamic= True)

train_df[start_index:end_index][['sales', 'forecast']].plot(figsize=(12, 8))

smape_kun(train_df[start_index:end_index]['sales'],train_df[start_index:end_index]['forecast'])