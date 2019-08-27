# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:59:23 2019

@author: dxuser22
"""

from heapq import heappush, heappop, heapify

ht = []
raw = []
for i in range(10):
    heappush(ht, i)
    raw.append(i)
    
for i in range(5, 0, -1):
    heappush(ht, i)
    raw.append(i)

print(ht)
print(raw)

ht2 = raw.copy()
heapify(ht2)
print(ht2)

for i in range(5):
    elem = heappop(ht2)
    print(elem, ht2)

