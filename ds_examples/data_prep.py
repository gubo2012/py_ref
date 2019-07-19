# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:51:12 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

def get_year_month(df, col):
    df[col] = df[col].astype(str)
    df['year'] = df[col].apply(lambda x:x[:4])
    df['month'] = df[col].apply(lambda x:x[4:6])
    return df
    

def remove_outliers(df, col):
    df=df[df[col] >= 0]
    y_mean = df[col].mean()
    y_std = df[col].std()
    df = df[df[col] <= y_mean + y_std * 6]
    return df


def normalize_store(df, col, col_y):
    df_groupby = df.groupby(col, as_index = False)[col_y].mean()
    df_groupby = df_groupby.rename(columns = {col_y:col_y + '_avg'})
    
    df = pd.merge(df, df_groupby, on = col, how = 'left')
    df[col_y] /= df[col_y + '_avg']
    
    df = df.drop(col_y + '_avg', axis=1)
    return df
    