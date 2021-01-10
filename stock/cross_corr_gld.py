# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:23:26 2021

@author: GuBo
"""

import pandas as pd

import ts_to_features

import seaborn as sns
import matplotlib.pyplot as plt

start_date = '2018-01-01'

shift_flag = True
shifts = 15

ticker = 'SLV'

df_spy = pd.read_pickle('spy.pkl')
df_qqq = pd.read_pickle('qqq.pkl')
df_ticker = pd.read_pickle('tickers.pkl')


# ETF
df_spy = df_spy[['date', 'Close']]
df_spy = df_spy.rename(columns = {'Close':'SPY'})
df_spy = ts_to_features.mongodb_format(df_spy)

df_qqq = df_qqq[['date', 'Close']]
df_qqq = df_qqq.rename(columns = {'Close':'QQQ'})
df_qqq = ts_to_features.mongodb_format(df_qqq)

df_ticker = df_ticker[['date', ticker]]
df_ticker = ts_to_features.mongodb_format(df_ticker)

# merge
df_merge = pd.merge(df_spy, df_qqq, how = 'inner', on = 'date')
df_merge = pd.merge(df_merge, df_ticker, how = 'inner', on = 'date')


# index to date
df = df_merge.copy()

#df.reset_index(level=0, inplace=True)


if shift_flag:
    for shift in range(shifts):
        df = ts_to_features.add_shift_cols(df, [ticker], shift+1)


df = df[df.date >= start_date]
end_date = df.date.max()

df.set_index('date', inplace=True, drop=True)
df.sort_values('date', inplace=True)


print('start_date {}, end_date {}'.format(start_date, end_date))
df_corr = df.corr()

print(df.corr())


df_plot = df.copy()
sns.lineplot(data=df_plot[['SPY', 'QQQ', ticker]])
#for tl in ax2.get_xticklabels():
#    tl.set_rotation(30)