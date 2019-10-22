# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:25:36 2019

@author: GuBo
"""

class Solution:
    def maxProduct(self, nums):
        
        prev_min = prev_max = global_max = nums[0]
        for num in nums[1:]:
            minn, maxx = min(num, prev_max*num, prev_min*num), max(num, prev_max*num, prev_min*num)
            prev_min, prev_max, global_max = minn, maxx, max(global_max, maxx)
        return global_max            
                
        

        
        
        
t1 = [2,3,-2,4]
print(Solution().maxProduct(t1))

t1 = [-2,0,-1]
print(Solution().maxProduct(t1))

t1 = [0,2]
print(Solution().maxProduct(t1))

t1 = [-2]
print(Solution().maxProduct(t1))

t1 = [-4, -3]
print(Solution().maxProduct(t1))

t1 = [-4, -3, -2]
print(Solution().maxProduct(t1))

t1 = [7,-2,-4]
print(Solution().maxProduct(t1))


t1 = [2,-5,-2,-4,3]
print(Solution().maxProduct(t1))