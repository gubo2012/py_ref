# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:27:18 2019

@author: dxuser22
"""

from collections import defaultdict

s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)

print(d.items())


# normal dict
d0 = {}
for k, v in s:
    d0[k].append(v)

print(d0.items())
