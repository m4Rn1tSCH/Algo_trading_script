# -*- coding: utf-8 -*-
"""
Created on 5/23/2020 11:26 PM

@author: bill-
"""
'''
This file contains all technical indicators for the stock data
'''
# TODO
# class still buggy

import matplotlib.pyplot as plt
from decouple import config
from alpha_vantage.techindicators import TechIndicators


class TechnicalIndicators:
    def __init__(self):
        self.api_key = config('AV_API_KEY')
        self.macd_data = self.macd()
        self.rsi_data = self.rsi()
        self.bbands_data = self.bbands()
        self.close_data = self.close()
        self.sma_data = self.sma()

    def macd(self):
        a = TechIndicators(key=self.api_key, output_format='pandas')
        data, meta_data = a.get_macd(symbol=self.stock_name,interval='daily')
        return data

    def rsi(self):
        a = TechIndicators(key=self.api_key,output_format='pandas')
        data,meta_data = a.get_rsi(symbol=self.stock_name,interval='daily',time_period=14)
        return data

    def bbands (self):
        a = TechIndicators(key=self.api_key,output_format='pandas')
        data,meta_data = a.get_bbands(symbol=self.stock_name)
        return data

    def sma(self):
        a = TechIndicators(key=self.api_key, output_format='pandas')
        data, meta_data = a.get_sma(symbol=self.stock_name,time_period=30)
        return data

    def wma(self):
        e = TechIndicators(key=self.api_key, output_format='pandas')
        data, meta_data = e.get_wma(symbol=self.stock_name, time_period=60)
        return data


if __name__ == "__main__":
    TI = TechnicalIndicators()
    close_data = TI.close_data
    macd_data = TI.macd_data
    rsi_data = TI.rsi_data
    bbands_data = TI.bbands_data
    sma_data = TI.sma_data
    plt.plot(macd_data)
    plt.show()
