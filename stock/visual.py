# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:35:07 2019

@author: GuBo
"""
import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns

# UDF
import stock_io

ticker = 'QQQ'

df_arima = pd.read_csv(stock_io.file_pred_low.format(ticker), header=None)
df_arima.columns = ['pred_ma']
df_lstm = pd.read_csv(stock_io.file_pred_high.format(ticker), header=None)
df_lstm.columns = ['pred_high']


df_actual = pd.read_csv(stock_io.format_data.format(ticker), header=0).tail(1500).reset_index(drop=True)
df_actual = df_actual[-252:].reset_index(drop=True)

df_comp = pd.concat([df_actual, df_arima, df_lstm], axis=1)
df_comp['pred'] = df_comp.pred_ma + df_comp.pred_high

#actual = data['close'].tail(252).values

df_plot = df_comp[['Date', 'close', 'pred_ma', 'pred']]

plt.figure(figsize = (16, 9))
sns.lineplot(x='Date', y='value', hue='variable', 
             data=pd.melt(df_plot, ['Date']))