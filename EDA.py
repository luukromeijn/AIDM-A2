# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 18:53:55 2022

@author: mag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import  scipy.interpolate as scip_in


data = pd.read_csv("sales_train.csv")
data['date'] = pd.to_datetime(data['date'],format = "%d.%m.%Y")



# EDA 
# Shop investigation

#How many items per shop?
item_count_per_shop = data.groupby("shop_id").nunique()['item_id']
item_count_per_shop.sort_values().plot.bar( figsize=(20,10),label= "Number of distinct items")
plt.legend()



# How many items sold in total per shop?

fig, ax = plt.subplots(figsize=(15,7))
item_sells_count_per_shop = data.groupby("shop_id").sum()['item_cnt_day']
ax.hlines( item_sells_count_per_shop.mean(), xmin=0, xmax=60, linewidth=2, color='r',label='mean number of sales')
item_sells_count_per_shop.sort_values().plot.bar( figsize=(20,10),label= "Number of sold items in total")
plt.legend()




# Seasonality

def seasonal_plot(X, y, period, freq, ax=None, legend = False):
    if ax is None:
        _, ax = plt.subplots()
        
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        legend = legend)
    ax.set_title(f"Seasonal Plot ({period}/{freq})")

    return ax

# Sales for all shops daily summed
sales_daily = data.groupby('date').sum()['item_cnt_day'].reset_index()

sales_daily['year'] = sales_daily['date'].dt.year
sales_daily['day_of_year'] = sales_daily['date'].dt.dayofyear

sales_daily['day_of_week'] = sales_daily['date'].dt.dayofweek
sales_daily['week'] = sales_daily['date'].dt.week


fig, ax = plt.subplots(figsize=(15,7))
seasonal_plot(sales_daily, y="item_cnt_day", period="week", freq="day_of_week")


high_values_yearly = sales_daily.groupby('year').quantile(q=0.98)['item_cnt_day']
high_dates = sales_daily[sales_daily.apply(lambda x:  x.item_cnt_day > high_values_yearly[x.year],axis=1)]

seasonal_plot(sales_daily, y="item_cnt_day", period="year", freq="day_of_year",ax=ax, legend = True)
ax.scatter(data=high_dates,x='day_of_year', y = 'item_cnt_day', c ='r',label='day of the high sales')

for idx, row in high_dates.iterrows():
    ax.annotate(str(row['day_of_year']), (row['day_of_year']+2, row['item_cnt_day'] + 4))

plt.legend()
plt.show()





sales_daily = sales_daily.sort_values(by='date')

fig, ax = plt.subplots(figsize=(15,7))

ax =sns.lineplot(x='date',y='item_cnt_day',data=sales_daily)
 
moving_average = sales_daily['item_cnt_day'].rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

moving_average = moving_average.reset_index()

moving_average['date'] = sales_daily['date']

ax = sns.lineplot(
        x='date',
        y='item_cnt_day',
        data=moving_average)
ax.set_title("365 day moving average ")


plt.legend()
plt.show()