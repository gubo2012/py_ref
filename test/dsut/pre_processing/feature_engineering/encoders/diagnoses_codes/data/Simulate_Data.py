# -*- coding: utf-8 -*-
"""
Code used to simulate a dataset with complicated
diagnoses and procedure codes

@author: Andy Wheeler
"""

import pandas as pd
import numpy as np
import os

my_dir = r'C:\Users\e009156\Documents\DataScience_Notes\PI_Project\Prelim_AmeriHealth_Scoring'
os.chdir(my_dir)

#I am going to simulate frequency and diagnoses codes
#About as often as they actually occur in the data
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
    #Returning this as a dataset
    return freq_set

#Generating simulated data as a test set
def sim_diag(code_list,n_cases,prob=None,maxd=10,flip=0.5):
    da_sim = []
    c_min = maxd-1
    for i in range(n_cases):
        #randomly choosing a size, minimum of 1
        rd = 1 + np.random.binomial(c_min, p=flip)
        #random selection without replacement
        rand_set = list(np.random.choice(code_list, size=rd, p=prob, replace=False))
        da_sim.append( rand_set + [np.nan]*(maxd-rd) )
    colm = ['d'+str(i+1) for i in range(maxd)]
    return pd.DataFrame(da_sim, columns=colm)

train_dt = pd.read_csv('SimpleDataset_AmeriHealth_Train.csv')
train_dt['Train'] = 1
test_dt = pd.read_csv('SimpleDataset_AmeriHealth_Test.csv')
test_dt['Train'] = 0
full_dat = train_dt.append(test_dt)
full_dat.describe()

di_col = ['diagCode'+str(i+1) for i in range(25)]
pc_col = ['procCode'+str(i+1) for i in range(6)]
code_freq = freq_tab(data=full_dat,fields=di_col)
proc_freq = freq_tab(data=full_dat,fields=pc_col)


sim_diags = sim_diag(code_list=code_freq.index, n_cases=len(full_dat.index), 
                     prob=code_freq['prop'], maxd=25)
#generate 1 extra and then drop the first column, since not all cases have a proc code
sim_prc = sim_diag(code_list=proc_freq.index, n_cases=len(full_dat.index), 
                   prob=proc_freq['prop'], maxd=7,flip=0.1)
sim_prc.drop(columns=['d1'],inplace=True)

sim_dat = full_dat.copy()

#For all of the other fields, permutate and add in
col_na = full_dat.columns.tolist()
ch_li = di_col + pc_col
for i in col_na:
    if i not in ch_li:
        sim_dat[i] = np.random.permutation(sim_dat[i])

sim_dat[di_col] = sim_diags
sim_dat[pc_col] = sim_prc
sim_dat = sim_dat[col_na]
print( sim_dat.head() )

sim_dat[sim_dat['Train'] == 1].to_csv('Simulated_Train.csv', index=False)
sim_dat[sim_dat['Train'] == 0].to_csv('Simulated_Test.csv', index=False)


