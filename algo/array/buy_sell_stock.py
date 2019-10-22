# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:03:58 2019

@author: GuBo
"""

class Solution:
    
    def maxProfit(self, prices) -> int:
        if len(prices) > 1:
            max_profit = 0
            p_min = prices[0]
            
            for p in prices:
                if p < p_min:
                    p_min = p
                else:
                    if p - p_min > max_profit:
                        max_profit = p - p_min

        else:
            max_profit = 0
        return max_profit
        
        
        

stock = Solution()
        
t1 = [7,1,5,3,6,4]
t1_profit = stock.maxProfit(t1)

t2 = [7,6,4,3,1]
t2_profit = stock.maxProfit(t2)

t3 = [2,1,2,1,0,1,2]
t3_profit = stock.maxProfit(t3)