# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:03:01 2019

@author: e009122
"""

import numpy as np

from datetime import datetime

date0 = '1/1/1900'
dt_date0 = datetime.strptime(date0, '%m/%d/%Y')

# convert a date to # of days from 1/1/1900
def dt_2_days(dt_str):
    if '/' in dt_str:
        dt_dt = datetime.strptime(dt_str.split(' ')[0], '%m/%d/%Y')
    else:
        dt_dt = datetime.strptime(dt_str.split(' ')[0], '%Y-%m-%d')
    dt_days = dt_dt - dt_date0
    return dt_days.days


# def conver dt col to days
def convert_dt_col(df, dt_col, date0='1900-1-1'):
#    dt_col = 'ELIGIBLE_START_DT'
    df[dt_col] = df[dt_col].astype(str)
    df[dt_col] = df[dt_col].fillna(date0)
    df.loc[df[dt_col] == 'nan', dt_col] = date0
    df.loc[df[dt_col] == '?', dt_col] = date0
    df[dt_col + '_days' ] = df[dt_col].apply(lambda x:dt_2_days(x))
    return df


def days_diff(df, col_start, col_end, col_diff):
    # make sure start dt is valid
    df[col_diff] = np.where(df[col_start]>0, df[col_end] - df[col_start], -1)
    # make sure end dt is valid
    df[col_diff] = np.where(df[col_end]>0, df[col_diff], -1)
    return df
    