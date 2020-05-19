# -*- coding: utf-8 -*-
"""
Created on 18 5/18/2020 8:15 PM 2020

@author: bill-
"""
from datetime import datetime as dt
import time
"""
-pull data every hour /every day
-prepare df and decide what to sell or buy
if loop -> buy or sell
append data every hour
calculate/or mark -> sell or buy order
"""
# TODO
# loop works
# pull one test stock
intra_df = pull_intraday_data(symbol='TSLA',
                            interval='5min',
                            outputsize='full',
                            output_format='pandas')

#add feature engineering columns that yield more accuracy
pred_feat(df=intra_df)
print(intra_df.head(10))

def test():
    while True:
        print("running")
        time.sleep(5)




