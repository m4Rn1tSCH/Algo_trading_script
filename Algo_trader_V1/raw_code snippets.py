# -*- coding: utf-8 -*-
"""
Created on 16 6/16/2020 4:43 PM

@author: bill-
"""
''' This file only contains only raw code that has not been used but might prove useful later'''

# intersection check for weighted moving averages with dates

# last_day_iso = dt.today() + timedelta(days=-1)
# next_day_iso = dt.today() + timedelta(days=1)
# yesterday_iso = last_day_iso.strftime('%Y-%m-%d')
# today_iso = dt.today().strftime('%Y-%m-%d')
# tomorrow_iso = next_day_iso.strftime('%Y-%m-%d')
# wma_50.get(yesterday_iso)
# wma_50.get(today_iso)
# wma_50.get(tomorrow_iso)

# Access of nested dictionary
# dict: (key): ((inner key, inner value))
for key, nested_value in wma_50.items():
    for wma_key, value in nested_value.items():
        # print the numerical inner value (the wma of a specific day)
        print(value)

for nested_dict in sorted(wma_200, reverse=True)[:10]:
    print(wma_200[nested_dict])

key_list = sorted(wma_50.keys(), reverse=True)[:3]
# last element for list slicing exclusive
for i, v in enumerate(key_list, 1):
    print("day-1:", wma_50[key_list[i + 1]])
    print("day:", wma_50[key_list[i]])
    print("day+1:", wma_50[key_list[i - 1]])