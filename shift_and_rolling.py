# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:17:52 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

import math

N = 200
scale = 10
x = np.random.rand(N, 2)

df = pd.DataFrame(x, columns = ['x1', 'x2'])


df['n'] = df.index

df['y1'] = df.x1*scale + df.n//scale + df.n.apply(lambda x:math.sin(x/5)*scale)
df['y2'] = df.x2*scale/2 + df.n * 2/scale + df.n.apply(lambda x:math.sin(2*x/5)*scale/2)


#df.plot(y = ['y1', 'y2'])

df_s = df.shift(2)
df_s = df_s[['y1', 'y2']]
df_s = df_s.rename(columns = {'y1':'y1_s', 'y2':'y2_s'})

df = pd.merge(df, df_s, left_index=True, right_index=True, how = 'left')


df_rolling = df.rolling(window=5).mean()
df_rolling = df_rolling[['y1', 'y2']]
df_rolling = df_rolling.rename(columns = {'y1':'y1_rl', 'y2':'y2_rl'})

df = pd.merge(df, df_rolling, left_index=True, right_index=True, how = 'left')


df.plot(y = ['y1', 'y1_s', 'y1_rl'])


#import matplotlib.pyplot as plt
#
#df_corr = df.corr()
#plt.matshow(df.corr())
#plt.show()

df_concise = df[['x1', 'x2', 'n', 'y1', 'y1_s', 'y1_rl']].copy()

df_concise['y1_1'] = df_concise['y1'] - df_concise['y1'].shift(1)
df_concise = df_concise[df_concise.y1_1 == df_concise.y1_1]
df_concise['y'] = df_concise.y1_1.apply(lambda x:1*(x>0))

y = df_concise['y'].values
X = df_concise[['x1', 'x2', 'n', 'y1_s', 'y1_rl', 'y1_1']].values


# split data into train and test sets
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
