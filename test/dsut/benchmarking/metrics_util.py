# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:58:05 2019

@author: e009122
"""

import pandas as pd
import numpy as np

from sklearn import metrics

def get_roc_data(y_actual, y_pred_prob, thresholds = []):
    # e.g., thresholds = [0.2, 0.5, 0.8]

    datas = []
    
    if thresholds == []:
        thresholds = range(1,100,1)
    else:
        thresholds = np.multiply(thresholds, 100.0)
        
        
    for THRESHOLD in thresholds:
        predictions=np.where(y_pred_prob > THRESHOLD/100.0, 1, 0)
        conf_matrix = metrics.confusion_matrix(y_actual, predictions)
        fallout = conf_matrix[0, 1] / conf_matrix[0].sum()
        data=[THRESHOLD/100.0, metrics.balanced_accuracy_score(y_actual,predictions), metrics.accuracy_score(y_actual, predictions), metrics.recall_score(y_actual, predictions),
                   metrics.precision_score(y_actual, predictions), metrics.roc_auc_score(y_actual, y_pred_prob), metrics.f1_score(y_actual,predictions,average='micro'),
                   fallout]
        datas.append(data)

    p_df = pd.DataFrame(data=datas, 
             columns=["threshold","balanced_accuracy",  "accuracy", "recall", "precision", "roc_auc_score",'f1_score', 'fallout'])
    
    return p_df
