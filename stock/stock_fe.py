# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:51:54 2021

@author: GuBo
"""

import pandas as pd
import numpy as np

import ts_to_features

def fe_pipeline(df, cols, total_shifts=10, mas=[20], scale_ma_flag=True, drop_col=False):
    
    if scale_ma_flag:
        # add MAs
        df = ts_to_features.add_mas(df, cols, mas)
        df = ts_to_features.add_ratio(df, cols, cols[0] + '_ma{}'.format(mas[0]))
    
    # multi shifts
    shift_cols = cols
    df = ts_to_features.add_multi_shifts(df, shift_cols, total_shifts)
    
    # add fake-date for forecasting
    df = ts_to_features.clone_last_row(df, shift_cols, days = 2)
    
    if drop_col:
        df = df.drop(cols, axis=1)
        if scale_ma_flag:
            df = df.drop([cols[0] + '_ma{}'.format(mas[0])], axis=1)
    
    return df


# process ratios
def add_pc_ratios(df):
    
    ratios_dict = {'EquityPC':'equity_pc.pkl',
                   'ETPPC':'etp_pc.pkl'}
    
    for key, value in ratios_dict.items():
        df_pc = add_pc_ratio(file_name=value, col_name=key)
        
        df = pd.merge(df, df_pc, how='left', on='date')
        
#        df = df.drop([key], axis=1)
    return df


def add_pc_ratio(file_name, col_name):
    
    df = pd.read_pickle(file_name)
    
    df = df[['date', 'PCRatio']]
    df = df.rename(columns = {'PCRatio':col_name})
    df = ts_to_features.mongodb_format(df)
    
    df = fe_pipeline(df, [col_name], scale_ma_flag = False, drop_col=True)
    
    return df
    
  
# process other tickers in ticker_list
def add_other_tickers(df, ticker_list):
    
    df_tickers = pd.read_pickle('tickers.pkl')
    df_tickers = ts_to_features.mongodb_format(df_tickers)
    
    for ticker in ticker_list:
        df_one_ticker = df_tickers[['date', ticker]]
        
        df_one_ticker = fe_pipeline(df_one_ticker, [ticker], drop_col=True)
        df = pd.merge(df, df_one_ticker, how='left', on='date')
        
    return df


# proces candle patterns
def add_cdl(df, df_cdl, patt_list, lag):
    
    cdl_list = patt_list.copy()
    cdl_list.append('date')
    df_cdl = df_cdl[cdl_list]
        
    df = pd.merge(df, df_cdl, how='left', on='date')

    for patt in patt_list:
        df[patt] = df[patt].shift(lag)

    return df
        

# add options        
def add_options(df, ticker):
    
    df_options = pd.read_pickle('options_all.pkl')
    df_options = df_options[df_options['symbol'] == ticker]
    
    cols_options = ['LTCallFlow', 'STCallFlow', 'LTPutFlow', 'STPutFlow']
    
    for col in cols_options:
        df_options[col] = df_options[col].fillna(0)
        df_options[col] = df_options[col] / df_options['Adj Close'] / df_options['Volume'] * 1000 * 1000
    
    df_options = df_options.drop(['symbol', 'CallFlow', 'PutFlow', 'Adj Close', 'Volume'], axis = 1)
        
    
    df_options = ts_to_features.mongodb_format(df_options)
    
    for col in cols_options:
        df_options[col] = df_options[col].fillna(0)
    
    df_options = fe_pipeline(df_options, cols_options, scale_ma_flag=False, drop_col=True)
    
    df = pd.merge(df, df_options, how='left', on='date')
    
      
    return df
    
    