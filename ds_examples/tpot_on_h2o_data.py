# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:20:57 2019

@author: dxuser22
"""

## Import a sample binary outcome train/test set into H2O
#train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
#test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

import pandas as pd
import numpy as np

from tpot import TPOTClassifier
import timeit

train = pd.read_csv('higgs_train_10k.csv')
test = pd.read_csv('higgs_test_5k.csv')

y_train = train['response']
y_test = test['response']

X_train = train.drop('response', axis=1).copy()
X_test = test.drop('response', axis=1).copy()

tpot = TPOTClassifier(verbosity=3, 
                      scoring="roc_auc", 
                      random_state=23, 
                      n_jobs=-1, 
                      generations=5, 
                      population_size=10)

times = []
scores = []
winning_pipes = []

# run three iterations and time them
for x in range(3):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_h2odata_pipeline.py')
times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)   
print('Winning pipelines:', winning_pipes)