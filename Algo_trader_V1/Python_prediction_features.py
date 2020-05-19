# -*- coding: utf-8 -*-
"""
Created on Sat April 22 10:26:42 2020

@author: bill-
"""
import numpy as np
from datetime import datetime as dt
# TODO
# NaNs that are being generated cannot be dropped properly
# find a way to properly drop those values and make data feasible

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

    #Conversion datetime to timestamps
    try:
        # handle datetime object and add features
        for col in list(df):
            if df[col].dtype == 'datetime64[ns]':
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_week"] = df[col].dt.week
                df[f"{col}_weekday"] = df[col].dt.weekday

        #conversion of dates to unix timestamps as numeric value (fl64)
        if df['date'].isnull().sum() == 0:
            df['date_ts'] = df['date'].apply(lambda x: dt.timestamp(x))
            df['date_ts'].astype('int64')
        else:
            df = df.drop(columns='date', axis=1)
            print("Column date dropped")
    except (TypeError, OSError, ValueError) as e:
        print(f"Problem with conversion:{e}")

    # typically engineered features based on lagging metrics
    # mean + stdev of past 3d/7d/30d/ + rolling volume
    df.reset_index(drop=True, inplace=True)
    # pick lag features to iterate through and calculate features
    lag_features = ['open', 'high', 'low', 'close', 'volume']
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
    try:
        df = df.drop(df.index[[0, 1]])
    except BaseException as e:
        print("ERROR OCCURED:", e)
    return df
