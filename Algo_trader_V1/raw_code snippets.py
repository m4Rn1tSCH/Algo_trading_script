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
#%%
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
#%%
# last element for list slicing exclusive
for i, v in enumerate(key_list, 1):
    for i, v in enumerate(key_list_2, 1):
        # access of nested dictionary
        # print("Date:", v, "WMA:", wma_200[v]['WMA'])
        print("Date -1:", key_list_2[i + 1], "WMA:", wma_200[key_list_2[i + 1]]['WMA'])
        print("Date:", key_list_2[i], wma_200[key_list_2[i]]['WMA'])
        print("Date +1:", key_list_2[i - 1], wma_200[key_list_2[i - 1]]['WMA'])
#%%
# generator statement to iterate over 2 lists
values = [(index50, index200, x, y) for index50, x in key_list for index200, y in key_list_2]
for index50, index200, x, y in values:
    print(index50, index200, x, y)
#%%
'''
Plot code
'''

# LINE VALUES
#   supported values are: '-', '--', '-.', ':',
#   'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
# Plot the date on x-axis and open price on y-axis

fig, ax = plt.subplots(2, 1, figsize=(15, 8))
plt.title('Open Price', style='oblique')
ax[0].plot(intra_df['date'], intra_df['open'], color='green', lw=1, ls='dashdot', marker=',', label="Open Price")
# Plot the date on x-axis and the trading volume on y-axis
#plt.set_title('Trading Volume', style='oblique')
ax[1].plot(intra_df['date'], intra_df['volume'], color='orange', lw=1, ls='--', marker='x', label="Trade Volume")
plt.title('Volume', style='oblique')

ax[0].legend(loc='upper right')
ax[1].legend(loc='upper right')
#
# plt.show()