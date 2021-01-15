# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:10:17 2021

@author: GuBo
"""

import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt


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
    
    next_date_fcst, y_test, y_pred = ml_score(df_test, model, fcst_day, fcst_day_total, target)
    
    #print('feature importance', model.feature_importances_)
    #plot_importance(model)
    
    df_features =  pd.DataFrame(
            {'feature': X_train.columns,
             'importance': model.feature_importances_})
    df_features = df_features.sort_values(by=['importance'], ascending = False)
        
    if target == 'target':
        print('*** Next date forecast: ', next_date_fcst)
    else:
        print('*** Next date forecast growth: %.2f%%' % (next_date_fcst*100))
    return df_features, next_date_fcst, df_test, y_test, y_pred
    
    

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
        print("In-Sample RMSE: {:.5f}".format(sqrt(mse)))
    
    return model, X_train


def ml_score(df_test, ml_model, fcst_day, fcst_day_total, target):
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
    
        # confusion matrix
        cm_labels = [-1, 0, 1]
        model_cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
        print('predicted = ', cm_labels)
        for i in range(3):
            print('Actual', cm_labels[i], model_cm[i], 'Sum', sum(model_cm[i]))
    #    y_test.hist()
    else:
        mse = mean_squared_error(y_test, y_pred)
        print("Out-of-Sample RMSE: {:.5f}".format(sqrt(mse)))
    
    return next_date_fcst, y_test, y_pred



