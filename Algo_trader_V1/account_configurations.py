# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:24:54 2019

@author: bill-
"""
#load necessary packages
import json as js
import requests as req
import acc_config

#%%
##retrieve account configurations
ENDPOINT_URL = "https://paper-api.alpaca.markets"
ACC_URL = "{}/v2/account".format(ENDPOINT_URL)
CONFIG_URL = "{}/v2/account/configurations".format(ENDPOINT_URL)
HEADERS = {'APCA-API-KEY-ID': acc_config.API_KEY, 'APCA-API-SECRET-KEY': acc_config.SECRET_KEY}
#%%
#can be turned into a dictionary
def get_acc():
    r_acc_info = req.get(ACC_URL, headers = HEADERS)
    #load is for a file-like object; loads is for is for strings;
    return js.loads(r_acc_info.content)

#retrieve acc config info
def get_acc_config():
    r_acc_info = req.get(CONFIG_URL, headers = HEADERS)
    #load is for a file-like object; loads is for is for strings;
    return js.loads(r_acc_info.content)

#%%
#update acc info
def update_acc():
    r_acc_update = req.patch(CONFIG_URL, headers = HEADERS)
    return js.loads(r_acc_update.content)

#returned dictionary object
#    {
#  "dtbp_check": "entry",
#  "no_shorting": false,
#  "suspend_trade": false,
#  "trade_confirm_email": "all"
#}