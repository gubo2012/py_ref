# -*- coding: utf-8 -*-
"""
Andy Wheeler

Plot to show binomial proportion confidence intervals

@author: e009156
"""

#Error bars for proportions 

import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
#import os

##################
#importing dataviz from ds-utilities folder
plot_code = r'C:\Users\e009156\Documents\GitHub\data-science-utils\plt'
import sys
sys.path.append(plot_code) #sys.path.append(".../")
import hms_plotstyle
hms_plotstyle.hms_plots()
##################


#Function to calculate upper/lower 95% confidence intervals
#For binary proportion, these are Clopper-Pearson exact CI's
def binom_interval(success, total, confint=0.95):
    quantile = (1 - confint) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return (np.nan_to_num(lower), np.where(np.isnan(upper), 1, upper))

agg_funs = [np.sum, len, np.mean]
ren_funs = {'sum':'Outcome',
            'len':'Total Cases',
            'mean':'Proportion'}

#Function to aggregate binary data into nice dataframe
def agg_bin(outcome, group, data, sort_low=True):
    grp = data.groupby(group, as_index=False)[outcome].agg(agg_funs).rename(columns=ren_funs)
    grp.rename(columns={'Outcome':outcome}, inplace=True)
    low, high = binom_interval(grp[outcome], grp['Total Cases'])
    grp['Low95CI'] = low
    grp['Hig95CI'] = high
    if sort_low:
        grp.sort_values(by='Low95CI', axis=0, ascending=False, inplace=True)
        grp.reset_index(inplace=True)
    if type(group) is not list:
        group_li = [group]
    else:
        group_li = group
    print( grp[ group_li + [outcome, 'Total Cases', 'Proportion', 'Low95CI'] ])
    return grp

#This is an error plot to illustrate differences in hypothetical samples
def error_plot(n, prop, conf=0.95, spike=True):
    #n = list(range(200,1050,50))
    #prop = [0.3]
    prop_df = pd.DataFrame(n, columns=['denom'])
    #Set up plot
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    for i, j in enumerate(prop):
        num_vec = j*prop_df['denom']
        prop_vec = [j for i in range(len(prop_df['denom']))]
        low, high = binom_interval(num_vec, prop_df['denom'], confint=conf)
        low_err = j - low
        high_err = high - j
        lab = 'Proportion ' + str(j)
        if spike:
            plt.errorbar(prop_df['denom'], prop_vec, yerr=[low_err, high_err], fmt='o', markeredgecolor='w', label=lab)
        else:
            ax.fill_between(prop_df['denom'], low, high, alpha=0.2, label=lab)
            ax.plot(prop_df['denom'], prop_vec, linestyle='-', marker='', markeredgecolor='w', label=lab)
    #Plot extras
    ax.set_title('Error Intervals around Proportions', fontsize=16)
    ax.set_xlabel('Sample Size')
    #plt.xticks(np.arange(200,1100,100))
    #plt.yticks(np.arange(0.23,0.38,0.01))
    #legend only if multiple proportions passed
    if len(prop) > 1:
        #works fine for spikes
        if spike:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        #more complicated for error areas
        else:
            handler, labeler = ax.get_legend_handles_labels()
            pl = len(prop)
            hd = [(handler[i],handler[i+pl]) for i in range(pl)]
            ax.legend(hd, labeler[0:pl], loc="center left", bbox_to_anchor=(1, 0.5))
    #plt.figtext(0.90, 0.07, 'Error bars are 95% Clopper-Pearson Confidence Intervals', horizontalalignment='right')
    #plt.savefig('FindingError_Time.png', dpi=500, bbox_inches='tight')
    plt.show()

#30% for sample sizes of 50
#error_plot(n = range(50,1050,50), prop = [0.3])

#More dense estimates, can then plot as a shaded background
#error_plot(n = range(50,1010,10), prop = [0.5], spike=False)
