# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:53:55 2019

@author: bill-
"""

#load necessary packages
import json as js
import requests as req
import acc_config
#%%
ENDPOINT_URL = "https://paper-api.alpaca.markets"
ACT_URL = "{}/v2/account/activities/{activity_type}".format(ENDPOINT_URL)
ALL_ACT_URL = "{}/v2/account/activities".format(ENDPOINT_URL)
HEADERS = {'APCA-API-KEY-ID': acc_config.API_KEY, 'APCA-API-SECRET-KEY': acc_config.SECRET_KEY}

#display one activity
def get_activity(activity_type, date, until, after):
    r_act = req.get(ACT_URL, headers = HEADERS)
    return js.loads(r_act.content)
#%%
#display one activity
def get_all_activity():
    r_all_act = req.get(ALL_ACT_URL, headers = HEADERS)
    return js.loads(r_all_act.content)