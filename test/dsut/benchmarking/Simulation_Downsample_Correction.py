# -*- coding: utf-8 -*-
"""
Simulation showing how to correct probabilities
After balancing the outcome sample for training

@author: Andy Wheeler
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

##############################################
#Simulate data with a low proportion of the outcome
np.random.seed(10)
tot_n = 1000000

sim_dat = pd.DataFrame({'Id': range(tot_n)})
sim_dat['X1'] = np.random.randn(tot_n)
sim_dat['X2'] = np.random.binomial(1,0.5,size=tot_n)
sim_dat['int_X1X2'] = sim_dat['X1']*sim_dat['X2']
sim_dat.head()

#Just change the intercept to a larger negative value to get a more rare outcome
sim_dat['Logit'] = -3.5 + 0.8*sim_dat['X1'] - 0.5*sim_dat['X2'] + 0.1*sim_dat['int_X1X2']
sim_dat['Prob'] = 1/(1 + np.exp(-sim_dat['Logit']))
sim_dat['Prob'].hist()
sim_dat['Y'] = np.random.binomial(1,sim_dat['Prob'])
print( sim_dat['Y'].mean() ) #Outcome only 3%

split = int(tot_n*0.8)
train_data = sim_dat[1:split]
test_data = sim_dat[split:tot_n]
##############################################

##############################################
#Take a stratified sample to get the proportion closer to 50%

out_data = train_data[train_data['Y'] == 1]
con_data = train_data[train_data['Y'] == 0]
con_samp = con_data.sample(n = len(out_data), replace=False)
#could also do weights, but this will be exact 50/50 split
sta_data = pd.concat([out_data, con_samp])
print( sta_data['Y'].mean() ) #Exact 50%
##############################################

##############################################
#Fit a random forest model

ind_vars = ['X1','X2','int_X1X2']
dep_var = 'Y'

#rf takes forever and is terrible
#only supplied X1 and X2, not interaction
#rf_mod = RandomForestClassifier(n_estimators=500, random_state=10)
rf_mod = LogisticRegression(penalty='none', solver='newton-cg')
rf_mod.fit(X = sta_data[ind_vars], y = sta_data[dep_var])

#View logit coefficients, see if they are unbiased
print( rf_mod.intercept_, rf_mod.coef_ )
#coefficients are close, but intercept is off
##############################################

##############################################
#Generate predictions for "new" testing data
pred_prob = rf_mod.predict_proba(test_data[ind_vars] )[::,1]

#Check the calibration with the predictions as is
ot, op = calibration_curve(test_data['Y'], pred_prob, n_bins=100, strategy='quantile')
##############################################

##############################################
#Use the formula to update probabilities
#And see if the calibration is correct
#https://github.com/matloff/regtools/blob/master/UnbalancedClasses.md

wrong_prop = sta_data['Y'].mean() #made 50% on purpose
true_prop = train_data['Y'].mean() #alittle over 3%

def adj_prop(cond_prob, wrong_p, true_p):
    wrong_ratio = (1 - wrong_p) / wrong_p
    #can get divide by zero errors for tree models
    fratio = (1/cond_prob - 1)*(1 / wrong_ratio)
    true_ratio = (1 - true_p) / true_p
    aprop = 1/(1 + true_ratio * fratio)
    #but those should be adjusted to 0
    return np.nan_to_num(aprop)

pred_adj = adj_prop(pred_prob, wrong_prop, true_prop) 

at, ap = calibration_curve(test_data['Y'], pred_adj, n_bins=100, strategy='quantile')

#Now make a nice graph superimposing those two
fix, ax = plt.subplots()
ax.plot(op, ot, label='Orig P', c='k')
ax.plot(ap, at, label='Adj P', c='r')
ax.legend()

#adjusted predictions for logistic are correct!
#if you do this with RF it is really bad
#no matter how you slice it
##############################################


##############################################
#This was extra stuff looking at RF
#Not sure why it was so bad with this data
#Pretty mild function and large sample

test_data['pred_prob'] = pred_prob
test_data['adj_prob'] = pred_adj

test_data['pred_prob'].hist()
test_data['adj_prob'].hist()

cal_bins = np.arange(0,1.01,0.01)
test_data['adj_bins'] = pd.cut(test_data['adj_prob'], cal_bins)
test_data['const'] = 1

adj_cal = test_data.groupby(['adj_bins'])[['Y','const']].sum()
adj_cal['act_prop'] = adj_cal['Y'] / adj_cal['const']
adj_cal['pred'] = np.arange(0,1,0.01) + 0.005

fix, ax = plt.subplots()
ax.scatter(adj_cal['pred'], adj_cal['act_prop'])
##############################################