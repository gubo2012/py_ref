import pandas as pd
import numpy as np

import talib
from talib import abstract

def get_functions():
    # Initialize moving averages from Ta-Lib, store functions in dictionary
    talib_moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA']
    functions = {}
    for ma in talib_moving_averages:
        functions[ma] = abstract.Function(ma)

    return talib_moving_averages, functions


def add_cdl(df, patt_list, lag_flag = False):
    
    for cdl_patt in patt_list:
        c_func = eval('talib.' + cdl_patt)
        if lag_flag:
            df[cdl_patt] = c_func(df['Open_lag1'].values, df['High_lag1'].values, df['Low_lag1'].values, df['Close_lag1'].values)
        else:
            df[cdl_patt] = c_func(df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values)
        
        df[cdl_patt] = df[cdl_patt] / 100
    return df
        

        
