# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:12:45 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np
#import seaborn as sns


def show_unique_value(df, col):
    n_unique = df[col].nunique()
    print(col, 'nunique:', n_unique)
    if n_unique <= 10:
        print(df[col].unique())


def get_mape(y_act, y_fcst):
    y_act, y_fcst = np.array(y_act), np.array(y_fcst)
    mape = np.abs(y_act - y_fcst).sum() / y_act.sum()
    return mape * 100