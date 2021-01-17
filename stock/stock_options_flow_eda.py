# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:53:23 2021

@author: GuBo
"""

import pandas as pd
import numpy as np

import util

ticker = 'W'

sof_file = '{}_options_flow.pkl'.format(ticker)

df_raw = pd.read_pickle(sof_file)


# a slice of df
df = df_raw.copy()
df = df[df['Date'] == '2021-01-13']


df_sum = df.groupby(['Expiry', 'OptionType']).agg({'Price': ['count', 'mean'], 'Strike': ['mean']})


## old method
#df2 = df.copy()
#df2 = df2[df2['Expiry'] == '2021-01-15']
#df2 = df2[df2['OptionType']=='C']

#
slice_dict = {'Expiry':'2021-01-15', 'OptionType':'C'}
df2 = util.slice_df(df.copy(), slice_dict)


df2_sum = df2.groupby(['Strike']).agg({'Price': ['count', 'mean'], 'Size': ['sum', 'mean', 'max']})


df3 = df2.copy()
df3 = df3[df3['Strike'] == 300]

df3 = df3[['Strike', 'NBBOBid', 'NBBOAsk', 'Size', 'Price', 'TradeIV', 'Flow']]
df3['AA'] = np.where(df3['Price']>=df3['NBBOAsk'], 1, 0)

size_50 = np.quantile(df3['Size'], 0.5)
size_75 = np.quantile(df3['Size'], 0.75)
size_90 = np.quantile(df3['Size'], 0.90)
size_95 = np.quantile(df3['Size'], 0.95)

df3['Size_gt_90'] = np.where(df3['Size'] >= size_90, 1, 0)
df3['Size_gt_95'] = np.where(df3['Size'] >= size_95, 1, 0)

# summary 1
df3_sum = df3.groupby(['AA', 'Size_gt_90', 'Size_gt_95']).agg({'Size': 'sum', 'Flow':'sum'})

sum_size_sum = df3_sum['Size'].sum()
sum_flow_sum = df3_sum['Flow'].sum()

df3_sum['Size_pct'] = df3_sum['Size'] / sum_size_sum
df3_sum['Flow_pct'] = df3_sum['Flow'] /sum_flow_sum

# summary 2
df3_sum2 = df3.groupby(['AA', 'Size_gt_90', 'Size_gt_95']).agg({'Size': {'sum', 'mean'}, 'Flow':{'sum', 'mean'}})