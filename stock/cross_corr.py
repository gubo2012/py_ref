# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:23:26 2021

@author: GuBo
"""

import pandas as pd

import ts_to_features

start_date = '2018-01-01'

shift_flag = True
shifts = 5

df_spy = pd.read_pickle('spy.pkl')
df_qqq = pd.read_pickle('qqq.pkl')
df_equity_pc = pd.read_pickle('equity_pc.pkl')
df_etp_pc = pd.read_pickle('etp_pc.pkl')


# ETF
df_spy = df_spy[['Adj Close']]
df_spy = df_spy.rename(columns = {'Adj Close':'SPY'})

df_qqq = df_qqq[['Adj Close']]
df_qqq = df_qqq.rename(columns = {'Adj Close':'QQQ'})


# PC Ratio
df_equity_pc = df_equity_pc[['PCRatio']]
df_equity_pc = df_equity_pc.rename(columns = {'PCRatio':'EquityPC'})

df_etp_pc = df_etp_pc[['PCRatio']]
df_etp_pc = df_etp_pc.rename(columns = {'PCRatio':'ETPPC'})


# merge
df_merge = pd.merge(df_spy, df_qqq, how = 'inner', on = 'date')
df_merge = pd.merge(df_merge, df_equity_pc, how = 'inner', on = 'date')
df_merge = pd.merge(df_merge, df_etp_pc, how = 'inner', on = 'date')


# index to date
df = df_merge.copy()

df.reset_index(level=0, inplace=True)


if shift_flag:
    for shift in range(shifts):
        df = ts_to_features.add_shift_cols(df, ['EquityPC', 'ETPPC'], shift+1)


df = df[df.date >= start_date]
end_date = df.date.max()

df.set_index('date', inplace=True, drop=True)
df.sort_values('date', inplace=True)


print('start_date {}, end_date {}'.format(start_date, end_date))
df_corr = df.corr()

print(df.corr())