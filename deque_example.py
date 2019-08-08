# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:57:32 2019

@author: dxuser22
"""

from collections import deque
import json

q = deque()

for i in range(10):
#    q.append(i)
    q.append((i, i*2, i * i, 'a'*i))
print(q)
    
for i in range(3):
    print(q.pop())
print(q)
    
for i in range(3):
    print(q.popleft())
print(q)


json_output = {}
json_output['q'] = list(q)

file_name = 'q_2_json.json'

with open(file_name, 'w') as f:
    json.dump(json_output, f)
    
    
with open(file_name) as f:
    q_dict_read = json.load(f)    

