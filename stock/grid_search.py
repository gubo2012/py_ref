# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:51:50 2021

@author: GuBo
"""

import json
import stock_io

def add_key_value(params, n, search_dict, search_key_list):
    key = search_key_list[n]
    for value in search_dict[key]:
        params[key] = value
        if n+1<len(search_key_list):
            add_key_value(params, n+1, search_dict, search_key_list)
        else:
            print(params)
            count = len(params_all)+1
            params_all[count] = params

search_dict = {}

search_dict['use_pc_flag'] = [0, 1]
search_dict['use_other_tickers'] = [0, 1]
search_dict['use_cdl_patt'] = [0, 1]
search_dict['use_short_vol_flag'] = [0, 1]

search_dict['ticker_list'] = [['GLD'], ['GLD', 'AGG'], ['GLD', 'SLV'], ['GLD', 'AGG', 'SLV']]

search_key_list = []
for key, value in search_dict.items():
    search_key_list.append(key)
    
    
params_all = {}

add_key_value({}, 0, search_dict, search_key_list)


with open(stock_io.grid_search, "w") as outfile:
    json.dump(params_all, outfile) 