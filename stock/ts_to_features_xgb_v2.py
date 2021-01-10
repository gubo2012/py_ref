# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:58:04 2021

@author: GuBo
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import stock_io
import ts_to_features
import ta_util
import util
import stock_ml
import stock_fe

ticker = 'QQQ'
up_down_threshold = 0.002 #0.2%
n_day_fcst = 1
total_shifts = 10

use_yahoo_flag = 0

use_pc_flag = 1
use_other_tickers = 1
#ticker_list = ['GLD', 'AGG']
ticker_list = ['GLD', 'AGG', 'SLV']
use_btc_flag = 0
use_cdl_patt = 1
patt_list = ['CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLENGULFING', 'CDLHARAMI', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLLONGLEGGEDDOJI', 'CDLMARUBOZU', 'CDLRICKSHAWMAN', 'CDLSHORTLINE']


if use_yahoo_flag:
    df = pd.read_csv(stock_io.raw_data.format(ticker))
else:
    df = pd.read_pickle(stock_io.pkl_data.format(ticker))
    df = ts_to_features.mongodb_format(df)

df = ts_to_features.data_format(df)

start_date = '2019-01-01'
test_date = '2020-08-01'
df = df[df.date >= start_date]

df_close = df.copy()
df_close = df_close[['date', 'Close']]

# use adj close instead of close
#df = df.drop(['Close'], axis=1)
#df = df.rename(columns = {'Adj Close':'Close'})
df = df.drop(['Adj Close'], axis=1)

df = df.sort_values(by=['date'])


df_raw_copy = df.copy()


# start feature engineering


df['CO_HL'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
df['HC_HL'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])

shift_only_cols = ['CO_HL', 'HC_HL']


# add candle patterns
if use_cdl_patt:
    df_cdl = df_raw_copy.copy()
    df_cdl = ta_util.add_cdl(df_cdl, patt_list)


# add MAs
df = ts_to_features.add_mas(df, ['Close'])
df = ts_to_features.add_mas(df, ['Volume'], [20])
    

# normalize
df['Close_raw'] = df['Close']
df = ts_to_features.add_ratio(df, ['Open', 'High', 'Low', 'Close', 'Close_ma10'], 'Close_ma20')
df = ts_to_features.add_ratio(df, ['Volume'], 'Volume_ma20')


#
## single shift
##df = ts_to_features.add_shift_cols(df, shift_cols, 1)

# multi shifts
shift_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_ma10', 'CO_HL', 'HC_HL']
df = ts_to_features.add_multi_shifts(df, shift_cols, total_shifts)






# add fake-date for forecasting

df = ts_to_features.clone_last_row(df, shift_cols, days = 2)


# add target
df = ts_to_features.add_shift_cols(df, ['Close_raw'], 1)
df['target'] = 0

df['target'] = np.where(df['Close_raw'] >= df['Close_raw_lag1d'] * (1+up_down_threshold), 1, df['target'])
df['target'] = np.where(df['Close_raw'] <= df['Close_raw_lag1d'] * (1-up_down_threshold), -1, df['target']) 


# for ts debug's purpose
#df_debug = df[['date', 'Close', 'Close_lag0d', 'Close_lag1d', 'Close_lag2d', 'Close_lag3d', 'Close_raw', 'Close_raw_lag1d', 'target']]


# ML
drop_list = ['Open', 'High', 'Low', 'Close', 'Volume',
             'CO_HL', 'HC_HL', 'Close_ma10', 'Close_ma20', 'Volume_ma20',
             'Close_raw', 'Close_raw_lag1d']
lag0d_list = util.show_cols(df, 'lag0d')
drop_list += lag0d_list

df = df.drop(drop_list, axis=1)




if use_pc_flag:
    df = stock_fe.add_pc_ratios(df)

if use_other_tickers:
    df = stock_fe.add_other_tickers(df, ticker_list)
#
#if use_btc_flag:
#    df = ts_to_features.add_btc(df)
#    
#
print('Ticker: ', ticker)

# 1-day fcst
print('1-day Forecast:')
if use_cdl_patt:
    df_1d = stock_fe.add_cdl(df.copy(), df_cdl.copy(), patt_list, lag = 1)
    
df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df_1d, test_date, 1, 2)
print(df_features.head(10))


# 2-day fcst
print('2-day Forecast:')
cols_lag1d = util.show_cols(df, 'lag1d')

df_2d = df.copy()
df_2d = df_2d.drop(cols_lag1d, axis=1)
if use_cdl_patt:
    df_2d = stock_fe.add_cdl(df_2d, df_cdl.copy(), patt_list, lag = 2)
    
df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df_2d, test_date, 2, 2)
print(df_features.head(10))


# backtest
cash_0 = 10000.0

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
        
    
            
         

##df_plot = pd.DataFrame({'date':df_test['date'][:-1], 'y_act':y_test, 'y_fcst':y_pred})
##df_plot = pd.merge(df_plot, df_close, on = 'date', how='left')
##df_plot['date'] = df_plot['date'].apply(lambda x:x[2:])
##df_plot.set_index('date', inplace=True, drop=True)
##
###%matplotlib qt
##sns.lineplot(data=df_plot['Close'], color = 'g')
##ax2 = plt.twinx()
##sns.lineplot(data=df_plot[['y_act', 'y_fcst']], ax=ax2)
###for tl in ax2.get_xticklabels():
###    tl.set_rotation(30)
#
#
