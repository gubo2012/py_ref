# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:09:56 2019

@author: gubo
"""

def knapsack(W, wt, val, n):
#    W, capacity left
#    n, available items
#    wt and val are arrays
    
    if n == 0 or W == 0:
        return 0
    
    if (wt[n-1] > W):
        return knapsack(W, wt, val, n-1)
    
    else:
#        take wt[n-1]
        option_1 = val[n-1] + knapsack(W - wt[n-1], wt, val, n-1)

#        not take wt[n-1]
        option_2 = knapsack(W, wt, val, n-1)
        return max(option_1, option_2)
    


val = [60, 100, 120] 
wt = [10, 20, 30] 
W = 50
n = len(val) 
print( knapsack(W , wt , val , n)  )

val = [60, 80, 100, 110, 120] 
wt = [10, 15, 20, 25, 30]
W = 65 
n = len(val) 
print( knapsack(W , wt , val , n)  )
