# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:56:36 2019

@author: e009122
"""

def drop_na(df):
    drop_list = []
    for col in df.columns:
        if df[col].nunique()<=1:
            drop_list.append(col)
    df = df.drop(drop_list, axis=1)
    return df


def check_drop(df, list_drop):
    list_drop_adj = []
    for col in list_drop:
        if col in df.columns:
            list_drop_adj.append(col)
    df = df.drop(list_drop_adj, axis=1)
    return df            


def get_first_3(df, col_input, col_output):
    df[col_output] = df[col_input].apply(lambda x:x[:3])
    return df


def get_unique_codes(df, col):
    df = df[df[col] == df[col]]
    
    col_unique = list(df[col].unique())
    col_unique_lst = list(map(lambda x:x.split('|'), col_unique))

    flat_list = [item for sublist in col_unique_lst for item in sublist]

    unique_codes = list(set(flat_list))
    return unique_codes
            
    
    