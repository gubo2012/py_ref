# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:03:51 2021

@author: GuBo
"""

import pandas as pd

def data_format(df):
    if 'Date' in df.columns:
        df = df.rename(columns = {'Date':'date'})
    if 'symbol' in df.columns:
        df = df.drop(['symbol'], axis=1)
    return df


def add_shift_cols(df, cols, shift, sum_flag = False):
    for col in cols:
        col_shift = col + '_lag{}'.format(shift)
        df[col_shift] = df[col].shift(shift)
        
        if sum_flag:
            df[col + '_sum'] += df[col_shift]
    return df


def ts_normalize(df, cols, shifts):
    for shift in range(shifts):
        for col in cols:
            col_shift = col + '_lag{}'.format(shift+1)
            df[col_shift] = df[col_shift] / df[col + '_avg']
    return df


def add_multi_shifts(df, cols, shifts):
    for col in cols:
        col_sum = col + '_sum'
        df[col_sum] = 0
    
    for shift in range(shifts):
        df = add_shift_cols(df, cols, shift+1, sum_flag = True)
    
    for col in cols:
        col_avg = col + '_avg'
        df[col_avg] = df[col + '_sum'] / shifts
        
    # remove all NA
    df = df[df[cols[0] + '_lag{}'.format(shifts)] == df[cols[0] + '_lag{}'.format(shifts)]]
    
    return df


def add_pc_ratios(df, shifts = 5):
    
    df_equity_pc = pd.read_pickle('equity_pc.pkl')
    df_etp_pc = pd.read_pickle('etp_pc.pkl')

    # PC Ratio
    df_equity_pc = df_equity_pc[['PCRatio']]
    df_equity_pc = df_equity_pc.rename(columns = {'PCRatio':'EquityPC'})
    df.reset_index(level=0, inplace=True)
    
    df_etp_pc = df_etp_pc[['PCRatio']]
    df_etp_pc = df_etp_pc.rename(columns = {'PCRatio':'ETPPC'})
    
    
    # merge
    df_merge = pd.merge(df, df_equity_pc, how = 'inner', on = 'date')
    df_merge = pd.merge(df_merge, df_etp_pc, how = 'inner', on = 'date')
    df.reset_index(level=0, inplace=True)
    
    for shift in range(shifts):
        df_merge = add_shift_cols(df_merge, ['EquityPC', 'ETPPC'], shift+1)
        
    df_merge = df_merge.drop(['EquityPC', 'ETPPC', 'index'], axis=1)
    
    return df_merge