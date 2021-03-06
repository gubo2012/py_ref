# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:47:33 2021

@author: GuBo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import stock_io
import ts_to_features
import ta_util
import util
import stock_ml
import stock_fe
import stock_bt

import json
import pickle
from functools import reduce

from config_manager import ConfigManager

conf_man = ConfigManager('./stock.yaml')


def analyze_ticker(ticker):
#    ticker = 'QQQ'
    
    pkl_file = open(stock_io.grid_search_output.format(ticker), 'rb')
    grid_search_results = pickle.load(pkl_file)
    pkl_file.close()
    
    
    df_flag_row0 = 1
    
    for key, value in grid_search_results.items():
        
        # convert list to str
        ticker_list = value['ticker_list']
        ticker_list_str = reduce((lambda x, y: x + '_' + y), ticker_list)
        value['ticker_list'] = ticker_list_str
        
    #    df_row = pd.DataFrame.from_dict(value, orient='index')
        df_row = pd.DataFrame([value])
        
        if df_flag_row0:
            df = df_row.copy()
            df_flag_row0=0
        else:
            df = df.append(df_row)
            
    
    
    n = 3
    conf_clf_threshold = 0.2
    df['overall_acc'] = 0
    df['overall_rmse'] = 0
    for i in range(n):
        df['overall_acc'] = df['overall_acc'] + df['{}d_clf_acc'.format(i+1)] / n
        df['overall_rmse'] = df['overall_rmse'] + df['{}d_reg_rmse'.format(i+1)] / n
        
    df['overall_acc_rank'] = df['overall_acc'].rank(ascending=False)
    df['overall_rmse_rank'] = df['overall_rmse'].rank(ascending=True)
    df['ranks_sum'] = df['overall_acc_rank'] + df['overall_rmse_rank']
    df['overall_rank'] = df['ranks_sum'].rank(ascending=True)
    
    df.sort_values('overall_rank', inplace=True)
    
    df_concise = df[['Ticker', 'ticker_list', 'use_cdl_patt', 'use_other_tickers', 'use_pc_flag',
                     'use_short_vol_flag', 'overall_acc', 'overall_rmse', 'overall_acc_rank',
                     'overall_rmse_rank', 'overall_rank']].copy()
    
    df_concise.sort_values('overall_rank', inplace=True)
#    print(df_concise.head())
    
    output_dict = df_concise[:5].to_dict('records')
    
    
    #emsemble
    emsemble_n = conf_man['emsemble_n']
    cols_fcst = util.show_cols(df, '_fcst')
    final_output = {'Ticker':ticker}
    final_output_confident = {'Ticker':ticker}
    df_top = df[:emsemble_n] # take top emsemble_n fcsts
    for col in cols_fcst:
        fcst = df_top[col].sum() / emsemble_n # take simple average
        if 'clf' in col:
            final_output[col] = round(fcst, 2)
        else:
            final_output[col] = '{}%'.format(round(fcst * 100, 1))
    
    # filter in only confident fcsts
    df_conf_output = pd.DataFrame([{'ticker':ticker}])
    
    for i in range(n):
        col_clf = f'{i+1}d_clf_fcst'
        col_reg = f'{i+1}d_reg_fcst'
        reg_fcst_num = float(final_output[col_reg][:-1])
        if final_output[col_clf] * reg_fcst_num > 0:
            if final_output[col_clf] > conf_clf_threshold or final_output[col_clf] < -conf_clf_threshold:
                final_output_confident[col_clf] = final_output[col_clf]
                final_output_confident[col_reg] = final_output[col_reg]
                df_conf_output[f'{i+1}d_fcst'] = float(final_output[col_reg][:-1]) # convert to float after remove %
            else:
                df_conf_output[f'{i+1}d_fcst'] = 0
        else:
            df_conf_output[f'{i+1}d_fcst'] = 0
        
    
#    # print flags
#    cols_flag = util.show_cols(df_top, 'use_')
#    flags = {}
#    for col in cols_flag:
#        flags[col] = round(df_top[col].mean(), 2)
#    print(ticker, flags)
#    
#    print(df_top['ticker_list'].mode())
    
    avg_acc = df_top['overall_acc'].mean()
    print('Avg Acc:', avg_acc, final_output)
    return final_output_confident, avg_acc, df_conf_output


if __name__ == '__main__':
    

    ticker_list = ['SPY', 'QQQ', 'BYND', 'W', 'TSLA', 'NIO', 'FUBO', 'BIDU', 'ARKK', 'MSFT', 'AMD', 'NVDA', 'AAL', 'GME', 'SLV', 'AMC',
                   'NOK', 'BA', 'BABA']
    
    confident_outputs = []
    avg_acc_outputs = {}
    
    
    #output to df to visualize
    summary_output_1st_flag = 1
    
    for ticker in ticker_list:
        confident_output, avg_acc, df_conf_output_row = analyze_ticker(ticker)
        if len(confident_output) > 1:
            confident_outputs.append(confident_output)
        avg_acc_outputs[ticker] = '{}%'.format(round(avg_acc * 100, 1))
        
        if summary_output_1st_flag:
            df_conf_output = df_conf_output_row.copy()
            summary_output_1st_flag = 0
        else:
            df_conf_output = df_conf_output.append(df_conf_output_row)
            
    print('Confident fcst only:', confident_outputs)
    print('1-3-day fcst avg accuracy:', avg_acc_outputs)
    
    n = 3
    df_chart = df_conf_output.copy()
    df_chart['day0'] = 1
    for i in range(n):
        df_chart['day{}'.format(i+1)] = df_chart['day{}'.format(i)] * (1 + df_chart['{}d_fcst'.format(i+1)] / 100)
    
        