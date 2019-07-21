# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:51:12 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

from datetime import datetime

def get_year_month(df, col):
    df[col] = df[col].astype(str)
    df['year'] = df[col].apply(lambda x:x[:4])
    df['month'] = df[col].apply(lambda x:x[4:6])
    
    df['dt'] = df[col].apply(lambda x:datetime.strptime(x, '%Y%m%d'))
    
    dt0 = df.dt.min()
    
    df['days'] = df['dt'].apply(lambda x: (x-dt0).days)
    
    return df
    

def remove_outliers(df, col):
    df=df[df[col] > 0]
    y_mean = df[col].mean()
    y_std = df[col].std()
    df = df[df[col] <= y_mean + y_std * 6]
    return df


def normalize_store(df, col, col_y):
    df_groupby = df.groupby(col, as_index = False)[col_y].mean()
    df_groupby = df_groupby.rename(columns = {col_y:col_y + '_avg'})
    df_groupby.to_pickle('stats.pkl')
    
    df = pd.merge(df, df_groupby, on = col, how = 'left')
    df[col_y] /= df[col_y + '_avg']
    
    df = df.drop(col_y + '_avg', axis=1)
    return df
    

def generate_df_test(df):
    df_test = df[df.DateStringYYYYMMDD == 20170714].copy()
    
    df_test.DateStringYYYYMMDD = 20170715
    df_test.Fiscal_dayofWk = df_test.Fiscal_dayofWk + 1
    df_test.Daypart = 'Lunch'
    df_test.HourlyWeather = 'clear-day'
    df_test.Hour = 12
    df_test.AvgHourlyTemp = 86
    df_test.SalesRevenue = 1
    
    df_test = df_test.drop_duplicates()
    return df_test
    

def add_ts(df, cols, col_y):
    cols_join = cols.copy()
    cols_join.append(col_y)
    
    df_m1 = df[cols_join].copy()
    df_m1.days = df_m1.days+1
    df_m1 = df_m1.rename(columns = {col_y:col_y + '_m1'})
    
    df_m7 = df[cols_join].copy()
    df_m7.days = df_m7.days+7
    df_m7 = df_m7.rename(columns = {col_y:col_y + '_m7'})
    
#    print('pre merge df len', len(df))
    df = pd.merge(df, df_m1, on = cols, how = 'left')
#    print('merge m1 df len', len(df))
    df = pd.merge(df, df_m7, on = cols, how = 'left')
#    print('merge m7 df len', len(df))
    
    df[col_y + '_m1'] = df[col_y + '_m1'].fillna(1)
    df[col_y + '_m7'] = df[col_y + '_m7'].fillna(1)
    
    return df
    

