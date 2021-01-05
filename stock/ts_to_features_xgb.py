# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:58:04 2021

@author: GuBo
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import stock_io
import ts_to_features

# ML
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score

def get_x_y(df):
    X = df.copy()
    X = X.drop(['date', 'target'], axis = 1)
    y = df['target']
    return X, y
 


ticker = 'SPY'
up_down_threshold = 0.005 #0.5%
n_day_fcst = 1
total_shifts = 15

use_yahoo_flag = 0

use_pc_flag = 1

if use_yahoo_flag:
    df = pd.read_csv(stock_io.raw_data.format(ticker))
else:
    df = pd.read_pickle(stock_io.pkl_data.format(ticker))
    df = ts_to_features.mongodb_format(df)

df = ts_to_features.data_format(df)

start_date = '2010-01-01'
test_date = '2020-08-01'
df = df[df.date >= start_date]

df_close = df.copy()
df_close = df_close[['date', 'Close']]

# use adj close instead of close
#df = df.drop(['Close'], axis=1)
#df = df.rename(columns = {'Adj Close':'Close'})
df = df.drop(['Adj Close'], axis=1)

df = df.sort_values(by=['date'])

shift_cols = ['Open', 'High', 'Low', 'Close', 'Volume']


# add fake-date for forecasting

#df_fake = df.copy()
#df_fake = df_fake.tail(1)
#df_fake.set_value(df_fake.index[-1], 'date', ts_to_features.fake_date)
#
#df = df.append(df_fake)

df = ts_to_features.clone_last_row(df)

# single shift
#df = ts_to_features.add_shift_cols(df, shift_cols, 1)

# multi shifts
df = ts_to_features.add_multi_shifts(df, shift_cols, total_shifts)


col_latest_close = 'Close'+ '_lag{}'.format(n_day_fcst)
df['target'] = 0

df['target'] = np.where(df['Close'] >= df[col_latest_close] * (1+up_down_threshold), 1, df['target'])
df['target'] = np.where(df['Close'] <= df[col_latest_close] * (1-up_down_threshold), -1, df['target']) 

df_copy = df.copy()
#df.target.hist()


df = ts_to_features.ts_normalize(df, shift_cols, total_shifts)

# ML
drop_list = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in shift_cols:
    drop_list.append(col + '_sum')
    drop_list.append(col + '_avg')

df = df.drop(drop_list, axis=1)

if use_pc_flag:
    df = ts_to_features.add_pc_ratios(df)


# ML pipeline
df_train = df[df.date < test_date]
df_test = df[df.date >= test_date]


# ML train
X_train, y_train = get_x_y(df_train)
X_test, y_test = get_x_y(df_test)


X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.20, random_state=7)

model = XGBClassifier()
eval_set = [(X_train_test, y_train_test)]
#model.fit(X_train_train, y_train_train, eval_metric='auc', eval_set=eval_set, verbose=True)
model.fit(X_train_train, y_train_train, eval_metric='auc')


print('Ticker: ', ticker)
if use_pc_flag:
    print('Use EquityPC & ETPPC')

# ML score
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print("In-Sample Accuracy: %.2f%%" % (accuracy * 100.0))

y_pred = model.predict(X_test)
next_date_fcst = y_pred[-1]

# remove the last row, fake_date, for accuracy cal
y_test = y_test[:-1]
y_pred = y_pred[:-1]

accuracy = accuracy_score(y_test, y_pred)
print("Out-of-Sample Accuracy: %.2f%%" % (accuracy * 100.0))

# confusion matrix
cm_labels = [-1, 0, 1]
model_cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
print('predicted = ', cm_labels)
for i in range(3):
    print('Actual', cm_labels[i], model_cm[i], 'Sum', sum(model_cm[i]))
y_test.hist()

#print('feature importance', model.feature_importances_)
#plot_importance(model)

df_features =  pd.DataFrame(
        {'feature': X_train.columns,
         'importance': model.feature_importances_})
df_features = df_features.sort_values(by=['importance'], ascending = False)
    

print('*** Next date forecast: ', next_date_fcst)

df_plot = pd.DataFrame({'date':df_test['date'][:-1], 'y_act':y_test, 'y_fcst':y_pred})
df_plot = pd.merge(df_plot, df_close, on = 'date', how='left')
df_plot['date'] = df_plot['date'].apply(lambda x:x[2:])
df_plot.set_index('date', inplace=True, drop=True)

#%matplotlib qt
sns.lineplot(data=df_plot['Close'], color = 'g')
ax2 = plt.twinx()
sns.lineplot(data=df_plot[['y_act', 'y_fcst']], ax=ax2)
#for tl in ax2.get_xticklabels():
#    tl.set_rotation(30)
