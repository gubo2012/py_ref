# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:57:05 2021

@author: GuBo
"""


import pandas as pd
import numpy as np

import datetime
import mplfinance as mpf

import ts_to_features
import stock_io

ticker = 'BYND'

df = pd.read_pickle(stock_io.stocks_all_data)
df = df[df['symbol'] == ticker]
df = ts_to_features.mongodb_format(df)


df_mpf = df.copy()

df_mpf.date = df_mpf.date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d'))
df_mpf.set_index('date', inplace=True, drop=True)

#mpf.plot(df_mpf,volume=True)


# add TA indicators
#apdict = mpf.make_addplot(df['LowerB'])
#
#mpf.plot(df,volume=True,addplot=apdict)



ap2 = [ mpf.make_addplot(df_mpf['short_vol_pct'],color='g',panel=2),  # panel 2 specified
      ]
mpf.plot(df_mpf,type='candle',style='charles', volume=True,addplot=ap2)