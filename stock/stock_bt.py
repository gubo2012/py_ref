# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:30:03 2021

@author: GuBo
"""

import pandas as pd

def run_backtest(df_raw_copy, df_test, y_pred, cash_0 = 10000.0):
    
    df_bt = df_test.copy()
    df_bt = df_bt[:-2]
    df_bt['pred'] = y_pred
    df_bt = df_bt[['date', 'pred']]
    
    df_bt['cash'] = cash_0
    df_bt['stock_value'] = 0.0
    df_bt['stock_share'] = 0.0
    df_bt = pd.merge(df_bt, df_raw_copy[['date', 'Open', 'Close']], how='left', on='date')
    
    for i in range(1, len(df_bt)):
        pred = df_bt.at[df_bt.index[i], 'pred']
        stock_share = df_bt.at[df_bt.index[i-1], 'stock_share']
        if pred == 1:
            if stock_share == 0:
                # buy
                cash = df_bt.at[df_bt.index[i], 'cash']
                stock_share = cash / df_bt.at[df_bt.index[i], 'Open']
                df_bt.at[df_bt.index[i], 'stock_share'] = stock_share
                df_bt.at[df_bt.index[i], 'stock_value'] = df_bt.at[df_bt.index[i], 'Close'] * stock_share
                df_bt.at[df_bt.index[i], 'cash'] = 0
            else:
                # do nothing
                df_bt.at[df_bt.index[i], 'cash'] = df_bt.at[df_bt.index[i-1], 'cash']
                df_bt.at[df_bt.index[i], 'stock_share'] = df_bt.at[df_bt.index[i-1], 'stock_share']
                df_bt.at[df_bt.index[i], 'stock_value'] = df_bt.at[df_bt.index[i], 'Close'] * df_bt.at[df_bt.index[i], 'stock_share']
        elif pred == -1:
            if stock_share > 0:
                # sell
                cash = df_bt.at[df_bt.index[i-1], 'stock_share'] * df_bt.at[df_bt.index[i], 'Open']
                df_bt.at[df_bt.index[i], 'stock_share'] = 0
                df_bt.at[df_bt.index[i], 'stock_value'] = 0            
            else:
                # do nothing
                df_bt.at[df_bt.index[i], 'cash'] = df_bt.at[df_bt.index[i-1], 'cash']
                df_bt.at[df_bt.index[i], 'stock_share'] = df_bt.at[df_bt.index[i-1], 'stock_share']
                df_bt.at[df_bt.index[i], 'stock_value'] = df_bt.at[df_bt.index[i], 'Close'] * df_bt.at[df_bt.index[i], 'stock_share']
        else:
            # do nothing
            df_bt.at[df_bt.index[i], 'cash'] = df_bt.at[df_bt.index[i-1], 'cash']
            df_bt.at[df_bt.index[i], 'stock_share'] = df_bt.at[df_bt.index[i-1], 'stock_share']
            df_bt.at[df_bt.index[i], 'stock_value'] = df_bt.at[df_bt.index[i], 'Close'] * df_bt.at[df_bt.index[i], 'stock_share']
    df_bt['port_value'] = df_bt['cash'] + df_bt['stock_value']
    stock_price_0 = df_bt.at[df_bt.index[1], 'Open']
    df_bt['hold_value'] = cash_0 * df_bt['Close'] / stock_price_0
    
    return df_bt