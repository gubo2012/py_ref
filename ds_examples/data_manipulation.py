# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 19:25:05 2019

@author: gubo
"""

import pandas as pd
import numpy as np

from scipy.stats import mode

df_raw = pd.read_csv('SalesbyHour.csv')

df = df_raw.sample(10000)

# mode (most frequent)
print('mode', mode(df.Daypart).mode[0])


# pivot and multiple index
df_pivot = df.pivot_table(values = 'SalesRevenue', index = ['Fiscal_Qtr', 'Daypart'],
                          aggfunc = np.mean)


# cross tab
df_crosstab = pd.crosstab(df.Hour, df.Daypart, margins=True)
#df_crosstab = pd.crosstab(df.Hour, df.Daypart)

def pct_convert(x):
#    return x/float(x[-1])
    return x/float(x[-1:][0])

df_crosstab_pct_col = df_crosstab.apply(pct_convert, axis=0)
df_crosstab_pct_row = df_crosstab.apply(pct_convert, axis=1)
