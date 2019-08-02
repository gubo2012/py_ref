# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:10:52 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np
 
 
header = pd.MultiIndex.from_product([['Semester1','Semester2'],['Maths','Science']])

d=([[12,45,67,56],[78,89,45,67],[45,67,89,90],[67,44,56,55]])
 
 
df = pd.DataFrame(d,
                  index=['Alisa','Bobby','Cathrine','Jack'],
                  columns=header)

stacked_df = df.stack()
stacked_df2 = df.stack().stack()

unstacked_df = stacked_df2.unstack()
unstacked_df2 = unstacked_df.unstack()

stacked_df_lvl0 = df.stack(level=0)

print(stacked_df.index.get_level_values(0))
print(stacked_df.index.get_level_values(1))

stacked_df['col_name'] = stacked_df.index.get_level_values(0)

df_reindex = stacked_df.copy()
df_reindex.reindex([('Bobby', 'Maths'), ('Jack', 'Science')])
