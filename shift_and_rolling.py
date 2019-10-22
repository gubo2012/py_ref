# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:17:52 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

import math

N = 100
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