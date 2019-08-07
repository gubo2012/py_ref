# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:30:34 2019

@author: dxuser22
"""

N = 8

# m, numbers filled already
def rules_pass(q_list, m):
    if m > 1:
        result = True
        i = 0
        while result and (i<m-1):
            j = i+1
            while result and (j<m):
                if q_list[i] == q_list[j]: result = False  # same column
                if result:
                    if abs(q_list[j]-q_list[i])==j-i: result = False  # diagonal
                j += 1
            i += 1
        return result
    else:
        return True
    

def solve_eq(q_list, n):    
    for j in range(N):
        q_list[N-n] = j
        if rules_pass(q_list, N-(n-1)):
            if n == 1: # last one
                print('solution', q_list)
            else:
                solve_eq(q_list.copy(), n-1)
         
q_list = [0] * N
solve_eq(q_list, N)
        