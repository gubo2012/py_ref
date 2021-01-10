# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:03:51 2021

@author: GuBo
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler

fake_date = '2029-12-0{}'
clone_days = 2

def data_format(df):
    if 'Date' in df.columns:
        df = df.rename(columns = {'Date':'date'})
    if 'symbol' in df.columns:
        df = df.drop(['symbol'], axis=1)
    return df

def mongodb_format(df):
#    df.reset_index(level=0, inplace=True)
    df['date']=df['date'].astype(str)
    return df


def add_shift_cols(df, cols, shift, sum_flag = False, scaler_flag = False):
    for col in cols:
        col_shift = col + '_lag{}d'.format(shift)
        df[col_shift] = df[col].shift(shift)
        
        if sum_flag:
            df[col + '_sum'] = df[col + '_sum'] + df[col_shift]
            
        if scaler_flag:
            scaler = StandardScaler()
            
            df[[col_shift]] = scaler.fit_transform(df[[col_shift]])
    return df


def ts_normalize(df, cols, shifts):
    for shift in range(shifts):
        for col in cols:
            col_shift = col + '_lag{}d'.format(shift+1)
            df[col_shift] = df[col_shift] / df[col + '_avg']
    return df


def remove_na(df, col):
    df = df[df[col] == df[col]]
    return df
    

def add_multi_shifts(df, cols, shifts, sum_flag=True):
    if sum_flag:
        for col in cols:
            col_sum = col + '_sum'
            df[col_sum] = 0
    
    for shift in range(shifts):
        df = add_shift_cols(df, cols, shift+1, sum_flag)
    
    if sum_flag:
        for col in cols:
            col_avg = col + '_avg'
            df[col_avg] = df[col + '_sum'] / shifts
        
    # remove all NA
#    df = df[df[cols[0] + '_lag{}d'.format(shifts)] == df[cols[0] + '_lag{}d'.format(shifts)]]
    df = remove_na(df, cols[0] + '_lag{}d'.format(shifts))
    
    return df


def add_pc_ratios(df, shifts = 5):
    
    df_equity_pc = pd.read_pickle('equity_pc.pkl')
    df_etp_pc = pd.read_pickle('etp_pc.pkl')

    # PC Ratio
    df_equity_pc = df_equity_pc[['date', 'PCRatio']]
    df_equity_pc = df_equity_pc.rename(columns = {'PCRatio':'EquityPC'})
    df_equity_pc = mongodb_format(df_equity_pc)
    df_equity_pc = clone_last_row(df_equity_pc, clone_days)
    
    
    df_etp_pc = df_etp_pc[['date', 'PCRatio']]
    df_etp_pc = df_etp_pc.rename(columns = {'PCRatio':'ETPPC'})
    df_etp_pc = mongodb_format(df_etp_pc)
    df_etp_pc = clone_last_row(df_etp_pc)
    
    # merge
    df_merge = pd.merge(df, df_equity_pc, how = 'inner', on = 'date')
    df_merge = pd.merge(df_merge, df_etp_pc, how = 'inner', on = 'date')
    df.reset_index(level=0, inplace=True)
    
    for shift in range(shifts):
        df_merge = add_shift_cols(df_merge, ['EquityPC', 'ETPPC'], shift+1)
    
    # remove all NA    
#    df_merge = df_merge[df_merge['EquityPC' + '_lag{}d'.format(shifts)] == df_merge['EquityPC' + '_lag{}d'.format(shifts)]]
    df_merge = remove_na(df_merge, 'EquityPC' + '_lag{}d'.format(shifts))
        
    df_merge = df_merge.drop(['EquityPC', 'ETPPC'], axis=1)
    
    return df_merge


def clone_last_row(df, days = clone_days):
    df_fake = df.copy()
    df_fake = df_fake.tail(1)
    
    for i in range(days):
        df_fake.set_value(df_fake.index[-1], 'date', fake_date.format(i+1))
        df = df.append(df_fake)
    return df


def add_other_tickers(df, shifts = 15):
    
    df_tickers = pd.read_pickle('tickers.pkl')
    df_tickers = mongodb_format(df_tickers)
    df_tickers = clone_last_row(df_tickers)

    ticker_list = [ticker for ticker in df_tickers.columns if ticker != 'date']
    
    # merge
    df_merge = pd.merge(df, df_tickers, how = 'inner', on = 'date')
    
    for shift in range(shifts):
        df_merge = add_shift_cols(df_merge, ticker_list, shift+1, scaler_flag = True)
    
    # remove all NA    
    df_merge = remove_na(df_merge, ticker_list[0] + '_lag{}d'.format(shifts))
        
    df_merge = df_merge.drop(ticker_list, axis=1)
    
    return df_merge


def add_btc(df, shifts = 2):
    
    df_btc = pd.read_pickle('btc.pkl')
    
    df_btc = df_btc[['date', 'price', 'volume_24h']]
    df_btc = df_btc.rename(columns = {'price':'BTC', 'volume_24h':'BTC_vol'})
    df_btc = mongodb_format(df_btc)
    df_btc = clone_last_row(df_btc)
    
    
    # merge
    df_merge = pd.merge(df, df_btc, how = 'inner', on = 'date')
    
    for shift in range(shifts):
        df_merge = add_shift_cols(df_merge, ['BTC', 'BTC_vol'], shift+1, scaler_flag = True)
    
    # remove all NA    
#    df_merge = df_merge[df_merge['EquityPC' + '_lag{}d'.format(shifts)] == df_merge['EquityPC' + '_lag{}d'.format(shifts)]]
    df_merge = remove_na(df_merge, 'BTC' + '_lag{}d'.format(shifts))
        
    df_merge = df_merge.drop(['BTC', 'BTC_vol'], axis=1)
    
    return df_merge


    