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


ticker = 'SPY'
up_down_threshold = 0.002 #0.2%
n_day_fcst = 1
total_shifts = 15

use_yahoo_flag = 0

use_pc_flag = 1
use_other_tickers = 1
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

shift_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

df['CO_HL'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
df['HC_HL'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])

shift_only_cols = ['CO_HL', 'HC_HL']

# add fake-date for forecasting

#df_fake = df.copy()
#df_fake = df_fake.tail(1)
#df_fake.set_value(df_fake.index[-1], 'date', ts_to_features.fake_date)
#
#df = df.append(df_fake)

df = ts_to_features.clone_last_row(df, days = 2)

# single shift
#df = ts_to_features.add_shift_cols(df, shift_cols, 1)

# multi shifts
df = ts_to_features.add_multi_shifts(df, shift_cols, total_shifts)
df = ts_to_features.add_multi_shifts(df, shift_only_cols, total_shifts, sum_flag = False)


col_latest_close = 'Close'+ '_lag{}d'.format(n_day_fcst)
df['target'] = 0

df['target'] = np.where(df['Close'] >= df[col_latest_close] * (1+up_down_threshold), 1, df['target'])
df['target'] = np.where(df['Close'] <= df[col_latest_close] * (1-up_down_threshold), -1, df['target']) 

df_copy = df.copy()
#df.target.hist()


df = ts_to_features.ts_normalize(df, shift_cols, total_shifts)

# ML
drop_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'CO_HL', 'HC_HL']
for col in shift_cols:
    drop_list.append(col + '_sum')
    drop_list.append(col + '_avg')

df = df.drop(drop_list, axis=1)

if use_pc_flag:
    df = ts_to_features.add_pc_ratios(df)

if use_other_tickers:
    df = ts_to_features.add_other_tickers(df)

if use_btc_flag:
    df = ts_to_features.add_btc(df)
    

print('Ticker: ', ticker)

# 1-day fcst
print('1-day Forecast:')
if use_cdl_patt:
    df = ta_util.add_cdl(df, patt_list, lag_flag = True, lag = 1)
    
df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df, test_date, 1, 2)
print(df_features.head(10))
df_1d = df.copy()

# 2-day fcst
print('2-day Forecast:')
cols_lag1d = util.show_cols(df, 'lag1d')
df = df.drop(cols_lag1d, axis=1)
if use_cdl_patt:
    df = ta_util.add_cdl(df, patt_list, lag_flag = True, lag = 2)
    
df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df, test_date, 2, 2)
print(df_features.head(10))

#
#df_plot = pd.DataFrame({'date':df_test['date'][:-1], 'y_act':y_test, 'y_fcst':y_pred})
#df_plot = pd.merge(df_plot, df_close, on = 'date', how='left')
#df_plot['date'] = df_plot['date'].apply(lambda x:x[2:])
#df_plot.set_index('date', inplace=True, drop=True)
#
##%matplotlib qt
#sns.lineplot(data=df_plot['Close'], color = 'g')
#ax2 = plt.twinx()
#sns.lineplot(data=df_plot[['y_act', 'y_fcst']], ax=ax2)
##for tl in ax2.get_xticklabels():
##    tl.set_rotation(30)


