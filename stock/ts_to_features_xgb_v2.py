# -*- coding: utf-8 -*-
"""
for one-time train/score/debug
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
import stock_bt


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


from config_manager import ConfigManager
conf_man = ConfigManager('./stock.yaml')


ticker = 'NOK'
print_features_flag = 1
n_day_fcst = 1



# load default settings
up_down_threshold = conf_man['up_down_threshold']
total_shifts = conf_man['total_shifts']
#up_down_threshold = 0.002 #0.2%
#total_shifts = 10


use_stocks_all_data = 1

use_pc_flag = 1
use_other_tickers = 1

#ticker_list = ['GLD', 'AGG']
#ticker_list = ['GLD', 'AGG', 'SLV']
#ticker_list = ['GLD', 'SLV']
ticker_list = ['GLD']
use_btc_flag = 0
use_cdl_patt = 1
use_short_vol_flag = 1
patt_list = ['CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLENGULFING', 'CDLHARAMI', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLLONGLEGGEDDOJI', 'CDLMARUBOZU', 'CDLRICKSHAWMAN', 'CDLSHORTLINE']

use_options = 1


if use_stocks_all_data:
    df = pd.read_pickle(stock_io.stocks_all_data)
    df = df[df['symbol'] == ticker]
    df = ts_to_features.mongodb_format(df)
else:
    df = pd.read_pickle(stock_io.pkl_data.format(ticker))
    df = ts_to_features.mongodb_format(df)

df = ts_to_features.data_format(df)

start_date = conf_man['train_start_date']
test_date = conf_man['test_start_date']
df = df[df.date >= start_date]

df_close = df.copy()
df_close = df_close[['date', 'Close']]

# use adj close instead of close
#df = df.drop(['Close'], axis=1)
#df = df.rename(columns = {'Adj Close':'Close'})

if use_short_vol_flag:
    df = df.drop(['Adj Close', 'ShortVolume'], axis=1)
else:
    df = df.drop(['Adj Close', 'ShortVolume', 'short_vol_pct'], axis=1)


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
if use_short_vol_flag:
    df = ts_to_features.add_mas(df, ['Volume', 'short_vol_pct'], [20])
else:
    df = ts_to_features.add_mas(df, ['Volume'], [20])
    

# normalize
df['Close_raw'] = df['Close']
df = ts_to_features.add_ratio(df, ['Open', 'High', 'Low', 'Close', 'Close_ma10'], 'Close_ma20')
df = ts_to_features.add_ratio(df, ['Volume'], 'Volume_ma20')
if use_short_vol_flag:
    df = ts_to_features.add_ratio(df, ['short_vol_pct'], 'short_vol_pct_ma20')


#
## single shift
##df = ts_to_features.add_shift_cols(df, shift_cols, 1)

# multi shifts
shift_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_ma10', 'CO_HL', 'HC_HL']
if use_short_vol_flag:
    shift_cols.append('short_vol_pct')
df = ts_to_features.add_multi_shifts(df, shift_cols, total_shifts)






# add fake-date for forecasting

df = ts_to_features.clone_last_row(df, shift_cols, days = 3)


# add target
df = ts_to_features.add_shift_cols(df, ['Close_raw'], 1)
df['target'] = 0

df['target'] = np.where(df['Close_raw'] >= df['Close_raw_lag1d'] * (1+up_down_threshold), 1, df['target'])
df['target'] = np.where(df['Close_raw'] <= df['Close_raw_lag1d'] * (1-up_down_threshold), -1, df['target']) 

df['target_reg'] = df['Close_raw'] / df['Close_raw_lag1d'] - 1
df = ts_to_features.remove_na(df, 'target_reg')

# for ts debug's purpose
#df_debug = df[['date', 'Close', 'Close_lag0d', 'Close_lag1d', 'Close_lag2d', 'Close_lag3d', 'Close_raw', 'Close_raw_lag1d', 'target']]


# ML
drop_list = ['Open', 'High', 'Low', 'Close', 'Volume',
             'CO_HL', 'HC_HL', 'Close_ma10', 'Close_ma20', 'Volume_ma20',
             'Close_raw', 'Close_raw_lag1d']
if use_short_vol_flag:
    drop_list.extend(['short_vol_pct', 'short_vol_pct_ma20'])
lag0d_list = util.show_cols(df, 'lag0d')
drop_list += lag0d_list

df = df.drop(drop_list, axis=1)




if use_pc_flag:
    df = stock_fe.add_pc_ratios(df)

if use_other_tickers:
    df = stock_fe.add_other_tickers(df, ticker_list)

if use_options:
    df = stock_fe.add_options(df, ticker)

#
#if use_btc_flag:
#    df = ts_to_features.add_btc(df)
#    
#
print('Ticker: ', ticker)
if use_short_vol_flag:
    print('Use short volume pct')



# 1 to 3 day fcst
output_dict = {'Ticker':ticker}
for i in range(3):
    n = i+1
    day_outout_dict = stock_ml.nth_day_fcst(df, df_cdl, n, patt_list, test_date, use_cdl_patt,
                                          print_features_flag=print_features_flag)
    output_dict.update(day_outout_dict)

print(output_dict)


## 1-day fcst
#print('1-day Forecast:')
#if use_cdl_patt:
#    df_1d = stock_fe.add_cdl(df.copy(), df_cdl.copy(), patt_list, lag = 1)
#    
#df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df_1d, test_date, 1, 2)
#if print_features_flag:
#    print(df_features.head(10))
#df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df_1d, test_date, 1, 2, target = 'target_reg')
#if print_features_flag:
#    print(df_features.head(10))
#
#
## 2-day fcst
#print('2-day Forecast:')
#cols_lag1d = util.show_cols(df, 'lag1d')
#
#df_2d = df.copy()
#df_2d = df_2d.drop(cols_lag1d, axis=1)
#if use_cdl_patt:
#    df_2d = stock_fe.add_cdl(df_2d, df_cdl.copy(), patt_list, lag = 2)
#    
#df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df_2d, test_date, 2, 2)
#if print_features_flag:
#    print(df_features.head(10))
#df_features, next_date_fcst, df_test, y_test, y_pred = stock_ml.ml_pipeline(df_2d, test_date, 2, 2, target = 'target_reg')
#if print_features_flag:
#    print(df_features.head(10))



# backtest
#df_bt = stock_bt.run_backtest(df_raw_copy, df_test, y_pred)
        
    
            
         

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

#import mplfinance as mpf
#import datetime
#df_mpf = df_raw_copy.copy()
##df_mpf.date = df_mpf.date.apply(datetime.datetime.fromtimestamp)
#df_mpf.date = df_mpf.date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d'))
#df_mpf.set_index('date', inplace=True, drop=True)
#mpf.plot(df_mpf, type='candle', mav=(3,6,9), volume=True)
