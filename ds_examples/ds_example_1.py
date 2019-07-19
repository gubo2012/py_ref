# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:08:39 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt 

import seaborn as sns

# UDF
import ds_util
import data_prep

boxplot_flag = False
regr_flag = False

df = pd.read_csv('SalesbyHour.csv')

df_sample = df.sample(1000)
print(df.columns)
df = df.rename(columns = {'SalesRevenue':'sales'})

print('pre removing outliers', df.sales.describe())
sns.distplot(df['sales'])


df = data_prep.remove_outliers(df, col = 'sales')
print('post removing outliers', df.sales.describe())
plt.figure()
sns.distplot(df['sales'])

df = data_prep.get_year_month(df, col = 'DateStringYYYYMMDD')



for col in df.columns:
    ds_util.show_unique_value(df, col)

cols_drop = ['DateStringYYYYMMDD']
df = df.drop(cols_drop, axis=1)


df = data_prep.normalize_store(df, 'Store_ID', 'sales')


cols_cat = ['Fiscal_Qtr', 'Fiscal_dayofWk', 'Daypart', 'HourlyWeather',
            'Hour', 'year', 'month']


if boxplot_flag:
    for col in cols_cat:
        plt.figure(figsize = (16, 9))
        sns.boxplot(x=col, y='sales', data=df)

# exclude 'Store_ID' 

df_cat = pd.get_dummies(df, columns = cols_cat)


if regr_flag:
    
    ## GLM
    y = df_cat['sales']
    X = df_cat.loc[:, ~df_cat.columns.isin(['sales', 'Store_ID'])]
    
    glm_model = sm.GLM(y, X, family = sm.families.Gaussian())
    glm_results = glm_model.fit()
    
    results_summary = glm_results.summary()
    print(results_summary)
    
    results_as_html = results_summary.tables[1].as_html()
    df_glm = pd.read_html(results_as_html, header=0, index_col=0)[0]
    df_glm['z_abs'] = np.abs(df_glm.z)
