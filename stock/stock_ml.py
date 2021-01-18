# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:10:17 2021

@author: GuBo
"""

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

import stock_fe
import test_util
import util

# ML
from xgboost import XGBClassifier, XGBRFRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix, roc_auc_score
from math import sqrt

def get_x_y(df, target):
    X = df.copy()
    X = X.drop(['date', 'target'], axis = 1)
    if 'target_reg' in X.columns:
        X = X.drop(['target_reg'], axis=1)
    y = df[target]
    return X, y
 
    
def ml_pipeline(df, test_date, fcst_day, fcst_day_total, target = 'target'):
    # ML pipeline
    df_train = df[df.date < test_date]
    df_test = df[df.date >= test_date]

    
    model, X_train = ml_train(df_train, target)
    
    next_date_fcst, y_test, y_pred, oos_benchmark = ml_score(df_test, model, fcst_day, fcst_day_total, target)
    
    #print('feature importance', model.feature_importances_)
    #plot_importance(model)
    
    df_features =  pd.DataFrame(
            {'feature': X_train.columns,
             'importance': model.feature_importances_})
    df_features = df_features.sort_values(by=['importance'], ascending = False)
        
    if target == 'target':
        print('****** Next {} day forecast: '.format(fcst_day), next_date_fcst)
    else:
        print('****** Next {} day forecast growth: %.2f%%'.format(fcst_day) % (next_date_fcst*100))
    return df_features, next_date_fcst, df_test, y_test, y_pred, oos_benchmark
    
    

def ml_train(df_train, target):
    
    X_train, y_train = get_x_y(df_train, target)
    # ML train
    
    X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.20, random_state=7)
    
    if target == 'target':
#        classification
        model = XGBClassifier()
        model.fit(X_train_train, y_train_train, eval_metric='mlogloss', eval_set=[(X_train_test, y_train_test)], early_stopping_rounds=25, verbose=False)
    else:
#        regression
        model = XGBRFRegressor()
        model.fit(X_train_train, y_train_train, eval_metric='rmse', eval_set=[(X_train_test, y_train_test)], early_stopping_rounds=25, verbose=False)        
    


    print('Training Set: {} to {}'.format(df_train['date'].min(), df_train['date'].max()))

    # ML score
    y_pred = model.predict(X_train)
    
    if target == 'target':
        accuracy = accuracy_score(y_train, y_pred)
        print("In-Sample Accuracy: %.2f%%" % (accuracy * 100.0))
    else:
        mse = mean_squared_error(y_train, y_pred)
        print("In-Sample RMSE: %.2f%%" % (sqrt(mse)*100))
    
    return model, X_train


def ml_score(df_test, ml_model, fcst_day, fcst_day_total, target,
             show_confusion_matrix_flag=0):
    fcst_day_offset = fcst_day_total + 1 - fcst_day # up to 2-day fcst
    X_test, y_test = get_x_y(df_test, target)
    
    y_pred = ml_model.predict(X_test)
    next_date_fcst = y_pred[- fcst_day_offset]

    # remove the last row, fake_date, for accuracy cal
    y_test = y_test[:- fcst_day_total]
    y_pred = y_pred[:- fcst_day_total]


    
    if target == 'target':
        
        accuracy = accuracy_score(y_test, y_pred)
        print("Out-of-Sample Accuracy: %.2f%%" % (accuracy * 100.0))
        oos_benchmark = accuracy
    
        if show_confusion_matrix_flag:
            # confusion matrix
            cm_labels = [-1, 0, 1]
            model_cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
            print('predicted = ', cm_labels)
            for i in range(3):
                print('Actual', cm_labels[i], model_cm[i], 'Sum', sum(model_cm[i]))
    #    y_test.hist()
    else:
        mse = mean_squared_error(y_test, y_pred)
        print("Out-of-Sample RMSE: %.2f%%" % (sqrt(mse)*100))
        oos_benchmark = sqrt(mse)
    
    return next_date_fcst, y_test, y_pred, oos_benchmark


def nth_day_fcst(df, df_cdl, n, patt_list, test_date,
                 n_total=3,
                 use_cdl_patt = 1, print_features_flag = 0):

    print('')
    print('    {}-day Forecast:'.format(n))    
    
    # n > 1-day fcst
    df_d = df.copy()
    if n>1:
        # remove 1 to n-1 day lags
        for i in range(n-1):
            cols_lag_p1d = util.show_cols(df, 'lag{}d'.format(i+1))        
            df_d = df_d.drop(cols_lag_p1d, axis=1)
    
    test_util.assert_no_prior_days_data(df_d, n)
    
    if use_cdl_patt:
        df_d = stock_fe.add_cdl(df_d.copy(), df_cdl.copy(), patt_list, lag = n)
    
    output_dict = {}    
    df_features, next_date_fcst, df_test, y_test, y_pred, oos_benchmark = ml_pipeline(df_d, test_date, n, n_total)
    output_dict['{}d_clf_acc'.format(n)] = round(oos_benchmark, 4)
    output_dict['{}d_clf_fcst'.format(n)] = next_date_fcst
    if print_features_flag:
        print(df_features.head(10))

    df_features, next_date_fcst, df_test, y_test, y_pred, oos_benchmark = ml_pipeline(df_d, test_date, n, n_total, target = 'target_reg')
    output_dict['{}d_reg_rmse'.format(n)] = round(oos_benchmark, 4)
    output_dict['{}d_reg_fcst'.format(n)] = round(next_date_fcst, 4)
    if print_features_flag:
        print(df_features.head(10))

    return output_dict


