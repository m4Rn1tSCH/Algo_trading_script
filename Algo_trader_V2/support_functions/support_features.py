# -*- coding: utf-8 -*-
"""
Created on Sat April 22 10:26:42 2020

@author: bill-
"""
from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def pred_feat(df):
    """
    FEATURE ENGINEERING
    This script adds datetime features to enhance accuracy of predictions
    especially for time series analysis packages like  FBPROPHET and KNN
    :return: engineered dataframe
    IMPORTANT
    The lagging features produce NaN for the first two rows due to unavailability
    of values
    NaNs need to be dropped to make scaling and selection of features working
    """

    # Conversion datetime to timestamps
    try:
        # handle datetime object and add features
        for col in list(df):
            if df[col].dtype == 'datetime64[ns]':
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_week"] = df[col].dt.week
                df[f"{col}_weekday"] = df[col].dt.weekday

        # conversion of dates to unix timestamps as numeric value (fl64)
        if df['date'].isnull().sum() == 0:
            df['date_ts'] = df['date'].apply(lambda x: dt.timestamp(x))
            df['date_ts'].astype('int64')
        else:
            df = df.drop(columns='date', axis=1)
            print("Column date dropped")
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion: {e}")

    # mean + stdev of past 3d/7d/30d/ + rolling volume
    df.reset_index(drop=True, inplace=True)
    # pick lag features to iterate through and calculate features
    lag_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # set up time frames; how many days/months back/forth
    t1 = 3
    t2 = 7
    t3 = 30
    # rolling values for all columns ready to be processed
    df_rolled_3d = df[lag_features].rolling(window=t1, min_periods=0)
    df_rolled_7d = df[lag_features].rolling(window=t2, min_periods=0)
    df_rolled_30d = df[lag_features].rolling(window=t3, min_periods=0)

    # calculate the mean with a shifting time window
    df_mean_3d = df_rolled_3d.mean().shift(periods=1).reset_index().astype(np.float32)
    df_mean_7d = df_rolled_7d.mean().shift(periods=1).reset_index().astype(np.float32)
    df_mean_30d = df_rolled_30d.mean().shift(periods=1).reset_index().astype(np.float32)

    # calculate the std dev with a shifting time window
    df_std_3d = df_rolled_3d.std().shift(periods=1).reset_index().astype(np.float32)
    df_std_7d = df_rolled_7d.std().shift(periods=1).reset_index().astype(np.float32)
    df_std_30d = df_rolled_30d.std().shift(periods=1).reset_index().astype(np.float32)

    for feature in lag_features:
        df[f"{feature}_mean_lag{t1}"] = df_mean_3d[feature]
        df[f"{feature}_mean_lag{t2}"] = df_mean_7d[feature]
        df[f"{feature}_mean_lag{t3}"] = df_mean_30d[feature]

        df[f"{feature}_std_lag{t1}"] = df_std_3d[feature]
        df[f"{feature}_std_lag{t2}"] = df_std_7d[feature]
        df[f"{feature}_std_lag{t3}"] = df_std_30d[feature]
    # IF SCALING IS NEEDED:
    # the first two rows of lagging values have NaNs which need to be dropped
    # drop the first and second row since the indicators refer to previous non-existent days
    # df = df.drop([0, 1])
    df.reset_index(drop=True, inplace=True)

    # drop NaNs to allow prediction models
    df = df.dropna()

    return df

# function produces NaNs in the df again!
def trading_support_resistance(df, bin_width=30):

    """
    Create indicator columns and add values
    :param sup_tol: supportive tolerance
    :param res_tol: resistance tolerance
    :param sup_count : support count
    :param res_count : resistance count
    :param sup : support
    :param res : resistance
    :param positions : positions open
    :param signal : signals found
    :return dataframe with columns added:
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
        df_section = df[x - bin_width: x + 1]

    support_level = min(df_section['Open'])
    resistance_level = max(df_section['Open'])
    range_level = resistance_level - support_level

    df['res'][x] = resistance_level
    df['sup'][x] = support_level
    # allow a 20% buffer back into the mean zone of the price movement
    df['sup_tol'][x] = support_level + 0.2 * range_level
    # allow a 20% buffer back into the mean zone of the price movement
    df['res_tol'][x] = resistance_level - 0.2 * range_level

    if df['Open'][x] >= df['res_tol'][x] and df['Open'][x] <= df['res'][x]:
        in_resistance += 1
        df['res_count'][x] = in_resistance
    elif df['Open'][x] <= df['sup_tol'][x] and df['Open'][x] >= df['sup'][x]:
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
    return df


def test_stationarity(timeseries, window=12, cutoff=0.01):
    """

    :param timeseries:
    :param window:
    :param cutoff:
    :return:
    """
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


def smape_kun(y_true, y_pred):
    """
    The function calculates the
    :param y_true: array, pd.Series; given targets of data set
    :param y_pred: array, pd.Series; predicted target values
    :return: float; MAPE and SMAPE
    """
    mape = np.mean(abs((y_true-y_pred) / y_true)) * 100
    smape = np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))).fillna(0))
    print('MAPE: %.2f %% \nSMAPE: %.2f'% (mape, smape), "%")
