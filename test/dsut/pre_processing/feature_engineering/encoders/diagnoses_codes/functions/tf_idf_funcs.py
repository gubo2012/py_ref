# -*- coding: utf-8 -*-
"""
Function to create and score a dataset
for TF-IDF based on procedure/diagnoses
codes

Andy Wheeler
"""

import pandas as pd
import numpy as np

def freq_tab(data,fields):
    #This assumes code does not appear twice in a particular claim
    #Creating the frequency/IDF matrix for claims from a table
    tot_rows = len(data.index)
    cnts = []
    for i in fields:
        cnts.append( data[i].value_counts() )
    cnt_data = pd.DataFrame(cnts)
    totals = cnt_data.sum()
    prop = totals/totals.sum()
    inv_log = np.log(tot_rows/totals)
    co = [1]*len(totals.index)
    freq_set = pd.DataFrame(zip(totals, prop, inv_log, co), 
               index=totals.index,
               columns=['Freq','prop','logIDF','const'])
    #I add in a nan row as a helper function for missing data encoding
    nan_row = pd.DataFrame(zip([0],[0],[-1],[1]), index=[np.nan], columns=['Freq','prop','logIDF','const'])
    #Returning this as a dataset
    return freq_set.append(nan_row)

#change NaN to '' if you want to count empty codes
#add function to do this same by groups

#Creating columns for IDF or one hot encoding based on
#the frequency table, limits to a particular N
#if you want, returns a dataframe of new columns
def encode_idf(base_data,code_names,freq_matrix,col_var=2,limit=5,stub=''):
    col_list = []
    temp_dat = base_data[code_names].copy()
    for index,row in freq_matrix.iterrows():
        if row[0] >= limit:
            col_name = index + stub
            col_list.append(col_name)
            bool_dat = base_data[code_names] == index
            temp_dat[col_name] = bool_dat.any(axis=1)*row[col_var]
    return temp_dat[col_list] 
#should I do something to return a function/pickle to score
#future datasets?

#Very similar to encode_idf, but instead it calculates a sum score of items that are rare
#might define rare by either frequency of item or proportion, here default to less than 
#1 in 5000 diagnoses codes
def rare_sum(base_data,code_names,freq_matrix,col_var='prop',less_than=1/5000):
    sub_mat = freq_matrix[pd.notna(freq_matrix.index)] #I dont want the missing columns here
    test_list = sub_mat.index[sub_mat[col_var] < less_than].tolist()
    return base_data[code_names].isin(test_list).sum(axis=1)
    
#This function will add in new rows to a frequency matrix based on
#Whether a new code is in a dataset, default missing values    
def add_indices(freq_matrix, score_data):
    old_ind = set(freq_matrix.index)
    new_ind = set(pd.unique(score_data.values.ravel('K')))
    up_ind = list(new_ind - old_ind)
    n_new = len(up_ind)
    n_items = pd.DataFrame(zip( [0]*n_new, [0]*n_new, [-1]*n_new,  [1]*n_new ), index=up_ind, columns=['Freq','prop','logIDF','const'])
    return freq_matrix.append(n_items)