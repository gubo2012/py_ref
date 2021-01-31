# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:58:04 2021

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

import logging
import json
import pickle
import sys
import warnings
from config_manager import ConfigManager


conf_man = ConfigManager('./stock.yaml')



logging.basicConfig(
        format = '%(asctime)-15s %(message)s',
        filename='grid_search.log',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

logger = logging.getLogger(__name__)


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def run_grid_search(ticker, params):
#    up_down_threshold = 0.002 #0.2%
#    total_shifts = 10
    up_down_threshold = conf_man['up_down_threshold']
    total_shifts = conf_man['total_shifts']
    
    
    use_stocks_all_data = 1
    
    use_pc_flag = params['use_pc_flag']
    use_other_tickers = params['use_other_tickers']
    use_cdl_patt = params['use_cdl_patt']
    use_short_vol_flag = params['use_short_vol_flag']
    use_options = params['use_options_flag']

    ticker_list = params['ticker_list']
    if ticker in ticker_list:
        ticker_list.remove(ticker)
    
    patt_list = ['CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLENGULFING', 'CDLHARAMI', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLLONGLEGGEDDOJI', 'CDLMARUBOZU', 'CDLRICKSHAWMAN', 'CDLSHORTLINE']
    
    print_features_flag = 0
    
    if use_stocks_all_data:
        df = pd.read_pickle(stock_io.stocks_all_data)
        df = df[df['symbol'] == ticker]
        df = ts_to_features.mongodb_format(df)
    else:
        df = pd.read_pickle(stock_io.pkl_data.format(ticker))
        df = ts_to_features.mongodb_format(df)
    
    df = ts_to_features.data_format(df)
    
    start_date = '2019-01-01'
#    start_date = '2020-03-01'
    test_date = '2020-10-01'
    df = df[df.date >= start_date]
    
    df_close = df.copy()
    df_close = df_close[['date', 'Close']]
    
    # use adj close instead of close
    #df = df.drop(['Close'], axis=1)
    #df = df.rename(columns = {'Adj Close':'Close'})
    
    if use_short_vol_flag:
        df = df.drop(['Adj Close', 'ShortVolume'], axis=1)
    else:
        df = df.drop(['Adj Close', 'ShortVolume', 'short_vol_pct'], axis=1)
    
    
    df = df.sort_values(by=['date'])
    
    
    df_raw_copy = df.copy()
    
    
    # start feature engineering
    
    
    df['CO_HL'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    df['HC_HL'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])
    
    shift_only_cols = ['CO_HL', 'HC_HL']
    
    
    # add options
    if use_options:
        df = stock_fe.add_options(df, ticker)
    
    
    # add candle patterns
    if use_cdl_patt:
        df_cdl = df_raw_copy.copy()
        df_cdl = ta_util.add_cdl(df_cdl, patt_list)
    else:
        df_cdl = pd.DataFrame({'empty' : []})
    
    
    # add MAs
    df = ts_to_features.add_mas(df, ['Close'])
    if use_short_vol_flag:
        df = ts_to_features.add_mas(df, ['Volume', 'short_vol_pct'], [20])
    else:
        df = ts_to_features.add_mas(df, ['Volume'], [20])
        
    
    # normalize
    df['Close_raw'] = df['Close']
    df = ts_to_features.add_ratio(df, ['Open', 'High', 'Low', 'Close', 'Close_ma10'], 'Close_ma20')
    df = ts_to_features.add_ratio(df, ['Volume'], 'Volume_ma20')
    if use_short_vol_flag:
        df = ts_to_features.add_ratio(df, ['short_vol_pct'], 'short_vol_pct_ma20')
    
    
    #
    ## single shift
    ##df = ts_to_features.add_shift_cols(df, shift_cols, 1)
    
    # multi shifts
    shift_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_ma10', 'CO_HL', 'HC_HL']
    if use_short_vol_flag:
        shift_cols.append('short_vol_pct')
    df = ts_to_features.add_multi_shifts(df, shift_cols, total_shifts)
    
    
    
    
    
    
    # add fake-date for forecasting
    
    df = ts_to_features.clone_last_row(df, shift_cols, days = 3)
    
    
    # add target
    df = ts_to_features.add_shift_cols(df, ['Close_raw'], 1)
    df['target'] = 0
    
    df['target'] = np.where(df['Close_raw'] >= df['Close_raw_lag1d'] * (1+up_down_threshold), 1, df['target'])
    df['target'] = np.where(df['Close_raw'] <= df['Close_raw_lag1d'] * (1-up_down_threshold), -1, df['target']) 
    
    df['target_reg'] = df['Close_raw'] / df['Close_raw_lag1d'] - 1
    df = ts_to_features.remove_na(df, 'target_reg')
    
    # for ts debug's purpose
    #df_debug = df[['date', 'Close', 'Close_lag0d', 'Close_lag1d', 'Close_lag2d', 'Close_lag3d', 'Close_raw', 'Close_raw_lag1d', 'target']]
    
    
    # ML
    drop_list = ['Open', 'High', 'Low', 'Close', 'Volume',
                 'CO_HL', 'HC_HL', 'Close_ma10', 'Close_ma20', 'Volume_ma20',
                 'Close_raw', 'Close_raw_lag1d']
    if use_short_vol_flag:
        drop_list.extend(['short_vol_pct', 'short_vol_pct_ma20'])
    lag0d_list = util.show_cols(df, 'lag0d')
    drop_list += lag0d_list
    
    df = df.drop(drop_list, axis=1)
    
    
    
    
    if use_pc_flag:
        df = stock_fe.add_pc_ratios(df)
    
    if use_other_tickers:
        df = stock_fe.add_other_tickers(df, ticker_list)
    #
    #if use_btc_flag:
    #    df = ts_to_features.add_btc(df)
    #    
    #
    print('Ticker: ', ticker)
    if use_short_vol_flag:
        print('Use short volume pct')
    
    
    
    # 1 to 3 day fcst
    output_dict = {'Ticker':ticker}
    for i in range(3):
        n = i+1
        day_outout_dict = stock_ml.nth_day_fcst(df, df_cdl, n, patt_list, test_date, use_cdl_patt,
                                                print_features_flag=print_features_flag)
        output_dict.update(day_outout_dict)
    
    print(output_dict)
    return output_dict


if __name__ == '__main__':
    

    ticker_list = ['SPY', 'QQQ', 'BYND', 'W', 'TSLA', 'NIO', 'FUBO', 'BIDU', 'ARKK', 'MSFT', 'AMD', 'NVDA', 'AAL']
    for ticker in ticker_list:
    
        
    #    ticker = 'QQQ'
        
        logger.info(f'Ticker {ticker} starts')
        
        final_output = {}
        
        with open(stock_io.grid_search) as json_file:
            params_all = json.load(json_file)
            
        count = 0
        for key, value in params_all.items():
            print('#{} params: '.format(count), value)
                  
            row_output = {'row':count}
            row_output.update(value)
            
            output_dict = run_grid_search(ticker, params=value)
            row_output.update(output_dict)
            
#            logger.info(f'Params #{count} done')
                       
            
            final_output[count]=row_output
            
            count += 1
            
        
    #    with open(stock_io.grid_search_output.format(ticker), "w") as outfile:
    #        json.dump(final_output, outfile) 
        
        output = open(stock_io.grid_search_output.format(ticker), 'wb')
        pickle.dump(final_output, output)
        output.close()
        
        logger.info(f'Ticker {ticker} done')