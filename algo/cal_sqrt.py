# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:16:01 2019

@author: gubo
"""

import random
import math

N = random.randint(1, 100000)

# guess
g = N * random.random()

def cal_error(N, g):
    return abs(N - g*g)

def cal_new_g(g):
    new_g = (g + N/g) / 2
    
    err1 = cal_error(N, (g + new_g)/2)
    err2 = cal_error(N, (new_g + N/g)/2)
    
    if err1 < err2:
        return (g + new_g)/2
    else:
        return (new_g + N/g)/2

print(N, math.sqrt(N))

for i in range(20):
    print(i, g, cal_error(N, g))   
    g = cal_new_g(g)  
    

