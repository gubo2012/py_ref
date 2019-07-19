# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:56:25 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

dates = pd.date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 4),
                  index=dates, columns=['A', 'B', 'C', 'D'])

df2 = pd.DataFrame(np.random.randn(8, 4),
                  index=dates, columns=['AA', 'BB', 'CC', 'DD'])

df_merged = pd.merge(df, df2, left_index=True, right_index=True)

df_ri = df.copy().reset_index()
df_ri = df_ri.rename(columns = {'index':'date'})



df2_ri = df2.copy().reset_index()
df2_ri = df2_ri.rename(columns = {'index':'date'})

df_merged2 = pd.merge(df_ri, df2_ri)


df3 = pd.concat([df2_ri, df2_ri], axis=0)
df3['AA'] = range(16)
df_merged3 = pd.merge(df_ri, df3)

df_merged4 = pd.merge(df_ri, df3, how = 'left')

df_merged5 = pd.merge(df_ri, df3, how = 'right')