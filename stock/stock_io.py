# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:26:22 2019

@author: GuBo
"""

import pandas as pd

stocks_all_data = 'stocks_all.pkl'
options_all_data = 'options_all.pkl'
ref_data = 'tickers.pkl'

pkl_data = '{}.pkl'

raw_data = '{}.csv'
format_data = '{}_format.csv'


file_pred_low = 'final_prediction_low_{}.csv'
file_pred_high = 'final_prediction_high_{}.csv'

grid_search = 'grid_search.json'

grid_search_output = '{}_gs_output.pkl'


def check_date_range(df, date_col='date'):
    print('date min {}, max {}'.format(df[date_col].min(), df[date_col].max()))

if __name__ == '__main__':
    
    # check data
    # stocks
    df = pd.read_pickle(stocks_all_data)
    print('Data: ', stocks_all_data)
    check_date_range(df)
    
    # options
    df = pd.read_pickle(options_all_data)
    print('Data: ', options_all_data)
    check_date_range(df)
    
    # reference data, e.g., GLD
    df = pd.read_pickle(ref_data)
    print('Data: ', ref_data)
    check_date_range(df)
    
    # ratios data
    for pc_data in ['equity_pc.pkl', 'etp_pc.pkl']:
        df = pd.read_pickle(pc_data)
        print('Data: ', pc_data)
        check_date_range(df)

    
    
    
    
    