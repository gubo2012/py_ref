# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:34:59 2019

@author: GuBo
"""

class Solution:
    def coinChange(self, coins , amount: int) -> int:
        
        stack = {0:0}
#        for i in range(amount):
#            stack[i+1] = float('inf')
        
        coins.sort(reverse=True)
        
        for c in coins:
            j = c
            while j <= amount:                
#                stack[j] = min(stack[j], stack[j - c] + 1) # work but slow
                if j not in stack:
                    if j-c in stack:
                        stack[j] = stack[j-c] + 1
                else:
                    if j-c in stack:
                        stack[j] = min(stack[j-c] + 1, stack[j])            
                j += 1

#        print(stack)        
        
#        if stack[amount] == float('inf'):
        if amount not in stack:
            return -1
        else:
            return stack[amount]
        

        
        
        
coins = [1, 2, 5]
amount = 11

#coins = [2]
#amount = 3

#coins = [1, 3, 5]
#amount = 8


#coins = [1,2147483647]
#amount = 2

#coins = [186,419,83,408]
#amount = 6249


print(Solution().coinChange(coins, amount))