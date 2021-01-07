# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:22:58 2021

@author: GuBo
"""

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

import talib
#from talib import abstract

# UDF
import stock_io
import ts_to_features




ticker = 'SPY'

use_yahoo_flag = 0


if use_yahoo_flag:
    df = pd.read_csv(stock_io.raw_data.format(ticker))
else:
    df = pd.read_pickle(stock_io.pkl_data.format(ticker))
    df = ts_to_features.mongodb_format(df)


df['SMA'] = talib.SMA(df['Close'])

#df['CDL2CROWS'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
#
## not working
##df['CDL3BLACKCROWS'] = abstract.Function('CDL3BLACKCROWS')(df['Open'], df['High'], df['Low'], df['Close'])
#
## alternative
#cdl_func = eval('talib.'+'CDL3BLACKCROWS')
#df['CDL3BLACKCROWS'] = cdl_func(df['Open'], df['High'], df['Low'], df['Close'])

 
#print(talib.get_functions())
#print(talib.get_function_groups())




# candles pattern stats
cdl_patts = talib.get_function_groups()['Pattern Recognition']
for cdl_patt in cdl_patts:
    c_func = eval('talib.' + cdl_patt)
    df[cdl_patt] = c_func(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
    
    print(cdl_patt, df[cdl_patt].unique())
    
