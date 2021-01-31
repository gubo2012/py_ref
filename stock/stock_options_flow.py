# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:53:23 2021

@author: GuBo
"""

import pandas as pd
import numpy as np

import util


trade_date = '2021-01-29'
excl_expiry = '2021-01-29' # exclude expiry
#excl_expiry = ''

def get_options_stats(df, quantile=0.9):
    df = df[['Strike', 'NBBOBid', 'NBBOAsk', 'Size', 'Price', 'TradeIV', 'Flow']]
    df['AA'] = np.where(df['Price']>=df['NBBOAsk'], 1, 0)
    df['BB'] = np.where(df['Price']<=df['NBBOBid'], 1, 0)
    large_size =  np.quantile(df['Size'], quantile)
    df['large_size_flag'] = np.where(df['Size'] >= large_size, 1, 0)
            
    df_sum = df.groupby(['large_size_flag', 'AA', 'BB']).agg({'Size': {'sum', 'mean'}, 'Flow':{'sum', 'mean'}})
    return df, df_sum  


def get_options_stats_dict(df, expiry, strike):
    
    stats_dict = {'Expiry':expiry, 'Strike':strike}
    
    # call options
    slice_dict = {'Expiry':expiry, 'OptionType':'C', 'Strike':strike}
    df2C = util.slice_df(df.copy(), slice_dict)
    
    if len(df2C) > 0:   
    
        df2C, df2C_sum = get_options_stats(df2C)
        
        try:
            call_bull_large_flow = df2C_sum.loc[(1, 1, 0), ('Flow', 'sum')] 
        except:
            call_bull_large_flow = 0
        try:
            call_bull_small_flow = df2C_sum.loc[(0, 1, 0), ('Flow', 'sum')] 
        except:
            call_bull_small_flow = 0
        try:
            call_bear_large_flow = df2C_sum.loc[(1, 0, 1), ('Flow', 'sum')] 
        except:
            call_bear_large_flow = 0
        try:
            call_bear_small_flow = df2C_sum.loc[(0, 0, 1), ('Flow', 'sum')] 
        except:
            call_bear_small_flow = 0
    
        stats_dict['call_bull_large_flow'] = round(call_bull_large_flow, 2)
        stats_dict['call_bull_small_flow'] = round(call_bull_small_flow, 2)
        stats_dict['call_bear_large_flow'] = round(call_bear_large_flow, 2)
        stats_dict['call_bear_small_flow'] = round(call_bear_small_flow, 2)
        
        stats_dict['call_flow_sum'] = round(df2C['Flow'].sum(), 2)
    else:
        stats_dict['call_bull_large_flow'] = 0
        stats_dict['call_bull_small_flow'] = 0
        stats_dict['call_bear_large_flow'] = 0
        stats_dict['call_bear_small_flow'] = 0       

        stats_dict['call_flow_sum'] = 0


    # put options
    slice_dict = {'Expiry': expiry, 'OptionType':'P', 'Strike':strike}
    df2P = util.slice_df(df.copy(), slice_dict)
    
    if len(df2P) > 0:        
    
        df2P, df2P_sum = get_options_stats(df2P)
        
        try:
            put_bear_large_flow = df2P_sum.loc[(1, 1, 0), ('Flow', 'sum')] 
        except:
            put_bear_large_flow = 0
        try:
            put_bear_small_flow = df2P_sum.loc[(0, 1, 0), ('Flow', 'sum')] 
        except:
            put_bear_small_flow = 0
        try:
            put_bull_large_flow = df2P_sum.loc[(1, 0, 1), ('Flow', 'sum')] 
        except:
            put_bull_large_flow = 0
        try:
            put_bull_small_flow = df2P_sum.loc[(0, 0, 1), ('Flow', 'sum')] 
        except:
            put_bull_small_flow = 0
        
        stats_dict['put_bear_large_flow'] = round(put_bear_large_flow, 2)
        stats_dict['put_bear_small_flow'] = round(put_bear_small_flow, 2)
        stats_dict['put_bull_large_flow'] = round(put_bull_large_flow, 2)
        stats_dict['put_bull_small_flow'] = round(put_bull_small_flow, 2)
        
        stats_dict['put_flow_sum'] = round(df2P['Flow'].sum(), 2)
    else:
        stats_dict['put_bear_large_flow'] = 0
        stats_dict['put_bear_small_flow'] = 0
        stats_dict['put_bull_large_flow'] = 0
        stats_dict['put_bull_small_flow'] = 0
        
        stats_dict['put_flow_sum'] = 0
        
    
    return stats_dict
    

ticker = 'GME'

sof_file = '{}_options_flow.pkl'.format(ticker)

df_raw = pd.read_pickle(sof_file)


# a slice of df
df = df_raw.copy()
df = df[df['Date'] == trade_date]

# exclude a certain expiry date
if excl_expiry != '':
    df = df[df['Expiry'] != excl_expiry]


#df_sum = df.groupby(['Expiry', 'OptionType']).agg({'Size': ['count', 'mean'], 'Strike': ['mean']})
#df_sum = df.groupby(['Expiry', 'Strike']).agg({'Size': ['sum', 'count', 'mean'], 'Flow': ['sum', 'count', 'mean']})

df_sort = df.groupby(['Expiry', 'Strike']).agg({'Size': 'sum'})
df_sort = df_sort.sort_values('Size', ascending=False) 
df_sort = df_sort.reset_index()

## call options
#slice_dict = {'Expiry':'2021-01-15', 'OptionType':'C', 'Strike':300}
#df2C = util.slice_df(df.copy(), slice_dict)
#
#df2C, df2C_sum = get_options_stats(df2C)
#
#call_bull_large_flow_pct = df2C_sum.loc[(1, 1, 0), ('Flow', 'sum')] / df2C['Flow'].sum()
#call_bull_small_flow_pct = df2C_sum.loc[(0, 1, 0), ('Flow', 'sum')] / df2C['Flow'].sum()
#call_bear_large_flow_pct = df2C_sum.loc[(1, 0, 1), ('Flow', 'sum')] / df2C['Flow'].sum()
#call_bear_small_flow_pct = df2C_sum.loc[(0, 0, 1), ('Flow', 'sum')] / df2C['Flow'].sum()
#
## put options
#slice_dict = {'Expiry':'2021-01-15', 'OptionType':'P', 'Strike':300}
#df2P = util.slice_df(df.copy(), slice_dict)
#
#df2P, df2P_sum = get_options_stats(df2P)
#
#put_bear_large_flow_pct = df2P_sum.loc[(1, 1, 0), ('Flow', 'sum')] / df2P['Flow'].sum()
#put_bear_small_flow_pct = df2P_sum.loc[(0, 1, 0), ('Flow', 'sum')] / df2P['Flow'].sum()
#put_bull_large_flow_pct = df2P_sum.loc[(1, 0, 1), ('Flow', 'sum')] / df2P['Flow'].sum()
#put_bull_small_flow_pct = df2P_sum.loc[(0, 0, 1), ('Flow', 'sum')] / df2P['Flow'].sum()
#
#
#stats_dict = {}
#stats_dict['call_bull_large_flow_pct'] = call_bull_large_flow_pct
#stats_dict['call_bull_small_flow_pct'] = call_bull_small_flow_pct
#stats_dict['call_bear_large_flow_pct'] = call_bear_large_flow_pct
#stats_dict['call_bear_small_flow_pct'] = call_bear_small_flow_pct
#
#stats_dict['put_bear_large_flow_pct'] = put_bear_large_flow_pct
#stats_dict['put_bear_small_flow_pct'] = put_bear_small_flow_pct
#stats_dict['put_bull_large_flow_pct'] = put_bull_large_flow_pct
#stats_dict['put_bull_small_flow_pct'] = put_bull_small_flow_pct

#expiry = '2021-02-05'
#strike = 300
#
##expiry = '2023-01-20'
##strike = 450
#
#stats_dict = get_options_stats_dict(df.copy(), expiry, strike)

print('Ticker {} trade_date {} most traded options by size:'.format(ticker, trade_date))


for i in range(10):
    expiry = df_sort.at[df_sort.index[i], 'Expiry']
    strike = df_sort.at[df_sort.index[i], 'Strike']
    stats_dict = get_options_stats_dict(df.copy(), expiry, strike)
    print('#{}: '.format(i+1))
    print(stats_dict)
    
    
    df_row = pd.DataFrame([stats_dict])
    df_row['rank'] = i + 1
    if i == 0:
        df_stats_output = df_row.copy()
    else:
        df_stats_output = df_stats_output.append(df_row)
            
df_stats_output['call_put_flow_ratio'] = df_stats_output['call_flow_sum'] / df_stats_output['put_flow_sum'] 
df_stats_output['bull_total'] = df_stats_output['call_bull_large_flow'] + df_stats_output['call_bull_small_flow'] + df_stats_output['put_bull_large_flow'] + df_stats_output['put_bull_small_flow']
df_stats_output['bear_total'] = df_stats_output['call_bear_large_flow'] + df_stats_output['call_bear_small_flow'] + df_stats_output['put_bear_large_flow'] + df_stats_output['put_bear_small_flow']
df_stats_output['bull_bear_flow_ratio'] = df_stats_output['bull_total'] / df_stats_output['bear_total'] 