# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:03:51 2021

@author: GuBo
"""

def add_shift_cols(df, cols, shift, sum_flag = False):
    for col in cols:
        col_shift = col + '_lag{}'.format(shift)
        df[col_shift] = df[col].shift(shift)
        
        if sum_flag:
            df[col + '_sum'] += df[col_shift]
    return df


def ts_normalize(df, cols, shifts):
    for shift in range(shifts):
        for col in cols:
            col_shift = col + '_lag{}'.format(shift+1)
            df[col_shift] = df[col_shift] / df[col + '_avg']
    return df


def add_multi_shifts(df, cols, shifts):
    for col in cols:
        col_sum = col + '_sum'
        df[col_sum] = 0
    
    for shift in range(shifts):
        df = add_shift_cols(df, cols, shift+1, sum_flag = True)
    
    for col in cols:
        col_avg = col + '_avg'
        df[col_avg] = df[col + '_sum'] / shifts
        
    # remove all NA
    df = df[df[cols[0] + '_lag{}'.format(shifts)] == df[cols[0] + '_lag{}'.format(shifts)]]
    
    return df