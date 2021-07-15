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
print(processed_df.head())
#####

# processed_df['date'] = pd.to_datetime(processed_df['date'], format="%Y-%m-%d")

# per 1 store, 1 item
# train_df = train[train['store']==1]
# train_df = train_df[train['item']==1]
# train_df = train_df.set_index('date')
# train_df['year'] = train['date'].dt.year
# train_df['month'] = train['date'].dt.month
# train_df['day'] = train['date'].dt.dayofyear
# train_df['weekday'] = train['date'].dt.weekday

sns.lineplot(x="date", y="Open",legend = 'full' , data=processed_df)
plt.show()

sns.lineplot(x="date", y="Volume",legend = 'full' , data=processed_df)
plt.show()

sns.boxplot(x="date", y="Open", data=processed_df)
plt.show()


train_df = processed_df[:2500]
train_df = train_df.set_index('date')
train_df['Close'] = train_df['Close'].astype(float)

train_df.head()

from statsmodels.tsa.seasonal import seasonal_decompose
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

# TODO
# check for viability
first_diff = train_df.Open - train_df.Open.shift(1)
first_diff = first_diff.dropna(inplace = False)
test_stationarity(first_diff, window = 12)

# external test; automatic minimization of auto regression
# from pmdarima import auto_arima
# stepwise_fit = auto_arima(df['AvgTemp'], trace=True,
# suppress_warnings=True)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_df.Open, lags=40, ax=ax1) #
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_df.Open, lags=40, ax=ax2)# , lags=40
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(first_diff, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(first_diff, lags=40, ax=ax2)
plt.show()

# Here we can see the acf and pacf both has a recurring pattern every 7 periods. Indicating a weekly pattern exists.
# Any time you see a regular pattern like that in one of these plots, you should suspect that there is some sort of
# significant seasonal thing going on. Then we should start to consider SARIMA to take seasonality into account

arima_mod6 = sm.tsa.ARIMA(train_df.Open, (6,1,0)).fit(disp=False)
print(arima_mod6.summary())


resid = arima_mod6.resid
print(normaltest(resid))
# returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
# the residual is not a normal distribution

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)
plt.show()

sns.distplot(resid ,fit = stats.norm, ax = ax0) # need to import scipy.stats
plt.show()
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(resid)

#Now plot the distribution using
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')

# analyze results
# ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arima_mod6.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arima_mod6.resid, lags=40, ax=ax2)


# consider seasonality by sarima
sarima_mod6 = sm.tsa.statespace.SARIMAX(train_df.Open, trend='n', order=(6,1,0)).fit()
print(sarima_mod6.summary())

resid = sarima_mod6.resid
print(normaltest(resid))

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)

sns.distplot(resid ,fit = stats.norm, ax = ax0) # need to import scipy.stats
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
fig = sm.graphics.tsa.plot_acf(arima_mod6.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arima_mod6.resid, lags=40, ax=ax2)
fig.show()

# TODO
# PREDICTION AND EVALUATION
# sarima needs all dates to be consecutive and filled with values
# use forward fill . ffill() to fill weekends and holidays
start_index = len(train_df)
end_index = len(processed_df)
train_df.reset_index(drop=False, inplace=True)
train_df.ffill()
train_df = train_df.set_index('date')
train_df['forecast'] = sarima_mod6.predict(start=start_index, end=end_index, dynamic=True)
train_df[start_index:end_index][['Open', 'forecast']].plot(figsize=(14, 10))
plt.show()


def smape_kun(y_true, y_pred):
    mape = np.mean(abs((y_true-y_pred)/y_true))*100
    smape = np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f'% (mape,smape), "%")

smape_kun(train_df[1730:1825]['sales'], train_df[1730:1825]['forecast'])

# SARIMAX - adding external variables
# per 1 store, 1 item
storeid = 1
itemid = 1
train_df = train[train['store']==storeid]
train_df = train_df[train_df['item']==itemid]

# train_df = train_df.set_index('date')
train_df['year'] = train_df['date'].dt.year - 2012
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.dayofyear
train_df['weekday'] = train_df['date'].dt.weekday

start_index = 1730
end_index = 1826

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