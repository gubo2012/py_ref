# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:08:39 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

from datetime import datetime

import statsmodels.api as sm
import matplotlib.pyplot as plt 

import seaborn as sns

# UDF
import ds_util
import data_prep

plot_flag = True
regr_flag = False

df_raw = pd.read_csv('SalesbyHour.csv')

#df_sample = df.sample(1000)
print(df_raw.columns)

df_test = data_prep.generate_df_test(df_raw)
df = pd.concat([df_raw, df_test], axis=0)

df = data_prep.get_year_month(df, col = 'DateStringYYYYMMDD')


print('pre removing outliers', df.SalesRevenue.describe())
if plot_flag:
    sns.distplot(df['SalesRevenue'])


df = data_prep.remove_outliers(df, col = 'SalesRevenue')
print('post removing outliers', df.SalesRevenue.describe())
if plot_flag:
    plt.figure()
    sns.distplot(df['SalesRevenue'])

    plt.figure(figsize = (16, 9))
#        plt.figure() 
    sns.boxplot(x='Store_ID', y='SalesRevenue', data=df)
        
#for col in df.columns:
#    ds_util.show_unique_value(df, col)



#cols_drop = ['DateStringYYYYMMDD']
#df = df.drop(cols_drop, axis=1)


df = data_prep.normalize_store(df, 'Store_ID', 'SalesRevenue')


df = data_prep.add_ts(df, cols=['Store_ID', 'days', 'Hour'],
            col_y='SalesRevenue')


cols_cat = ['Fiscal_Qtr', 'Fiscal_dayofWk', 'Daypart', 'HourlyWeather',
            'Hour', 'year', 'month']


if plot_flag:
    for col in cols_cat:
        plt.figure(figsize = (16, 9))
#        plt.figure() 
        sns.boxplot(x=col, y='SalesRevenue', data=df)
    plt.figure(figsize = (16, 9))
#    plt.figure()
    sns.scatterplot(x="AvgHourlyTemp", y="SalesRevenue", data=df)
    
    plt.figure(figsize = (16, 9))
    df_by_day = df.groupby('DateStringYYYYMMDD', as_index=False)['SalesRevenue'].mean()
    df_by_day = df_by_day[df_by_day.DateStringYYYYMMDD != '20170715']
    sns.lineplot(x='DateStringYYYYMMDD', y='SalesRevenue', data=df_by_day)

# exclude 'Store_ID' 


df_cat = pd.get_dummies(df, columns = cols_cat)

df_cat.to_pickle('sales.pkl')
