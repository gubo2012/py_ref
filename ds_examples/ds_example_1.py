# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:08:39 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

from datetime import datetime

import statsmodels.api as sm
import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.linear_model import Ridge

# UDF
import ds_util
import data_prep


def get_x_y(df_cat, col_y):
    y = df_cat[col_y]
    X = df_cat.loc[:, ~df_cat.columns.isin(cols_excl)]
    X = X.loc[:, ~X.columns.isin([col_y])]
    return X, y
    

def get_sales(df, y_fcst, df_stats):
    df['y_fcst'] = y_fcst
    df = pd.merge(df, df_stats)
    
    df['sales_act'] = df['SalesRevenue'] * df['SalesRevenue_avg']
    df['sales_fcst'] = df['y_fcst'] * df['SalesRevenue_avg']
    
    return df


def mape_by_day(df):
    
    df = df.groupby('days', as_index=False)['sales_act', 'sales_fcst'].sum()
    
    y = df['sales_act'].copy()
    y_fcst = df['sales_fcst'].copy()    
    
    mape = ds_util.get_mape(y, y_fcst)
    print('all stores per day mape', mape)
    
df_stats = pd.read_pickle('stats.pkl')

def score(y, y_fcst, df, df_stats=df_stats):
    mape = ds_util.get_mape(y, y_fcst)
    print('len {}, mape {}'.format(len(df), mape))
    
    df_restore = get_sales(df, y_fcst, df_stats)
    mape_by_day(df_restore)
    
    print('overall accuracy', (df_restore.sales_fcst.sum() / df_restore.sales_act.sum()))


regr_flag = True

df_cat = pd.read_pickle('sales.pkl')


cols_excl = ['Store_ID', 'DateStringYYYYMMDD', 'dt']

col_y = 'SalesRevenue'

days_split = 1100
df_cat_train = df_cat[df_cat.days <= days_split].copy()
df_cat_test = df_cat[df_cat.days > days_split].copy()
df_cat_test = df_cat_test[df_cat_test.days < 1293]
df_cat_final = df_cat[df_cat.days == 1293].copy()

X, y = get_x_y(df_cat, col_y)
X_train, y_train = get_x_y(df_cat_train, col_y)
X_test, y_test = get_x_y(df_cat_test, col_y)
X_final, y_final = get_x_y(df_cat_final, col_y)

if regr_flag:
    
    ## GLM
    
    glm_model = sm.GLM(y, X, family = sm.families.Gaussian())
    glm_results = glm_model.fit()
    
    results_summary = glm_results.summary()
    print(results_summary)
    
    results_as_html = results_summary.tables[1].as_html()
    df_glm = pd.read_html(results_as_html, header=0, index_col=0)[0]
    df_glm['z_abs'] = np.abs(df_glm.z)
    
    lr_model = Ridge()
    
    print('________ all-all ________')

    lr_model.fit(X, y)
    y_fcst = lr_model.predict(X)
    y_final_fcst = lr_model.predict(X_final)
    
    score(y, y_fcst, df_cat)
    
    
    print('________ train-train ________')
    lr_model.fit(X_train, y_train)
    y_fcst_train = lr_model.predict(X_train)
    
    score(y_train, y_fcst_train, df_cat_train)
    
    print('________ train-test ________')
    
    y_fcst_test = lr_model.predict(X_test)
    
    score(y_test, y_fcst_test, df_cat_test)
    
    
    
    
    df_final = get_sales(df_cat_final, y_final_fcst, df_stats)
    print('final fcst agg', df_final.sales_fcst.sum())
    
    df_final['sales_m1'] = np.where(df_final['SalesRevenue_m1'] == 1, 0, df_final.SalesRevenue_m1 * df_final.SalesRevenue_avg)
    df_final['sales_m7'] = np.where(df_final['SalesRevenue_m7'] == 1, 0, df_final.SalesRevenue_m7 * df_final.SalesRevenue_avg)
    
    df_final_summary = pd.melt(df_final, id_vars = ['Store_ID'],
                               value_vars = ['sales_fcst', 'sales_m1', 'sales_m7', 'SalesRevenue_avg'])
    
    plt.figure(figsize = (16, 9))
    df_final_summary = df_final_summary.rename(columns = {'value':'SalesRevenue',
                                                          'variable':'Legend'})
#    sns.barplot(x='Store_ID', y='value', hue='variable', data=df_final_summary)
    sns.barplot(x='Store_ID', y='SalesRevenue', hue='Legend', data=df_final_summary)

    
    
    
    
