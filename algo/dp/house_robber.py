# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:09:39 2019

@author: GuBo
"""

class Solution:
    def rob(self, nums) -> int:
        
        def robbed(i):
            if i+2 <= len(nums)-1:
                if i+2 not in self.stack:
                    self.stack[i+2] = robbed(i+2)
                option1 = nums[i+2] + self.stack[i+2]

                if i+3 <= len(nums) - 1:
                    if i+3 not in self.stack:
                        self.stack[i+3] = robbed(i+3)
                    option2 = nums[i+3] + self.stack[i+3]
                else:
                    option2 = 0
                return max(option1, option2)
            else:
                return 0
        
        self.stack = {}
        
        if len(nums) > 1:
            output = max(nums[0] + robbed(0), nums[1] + robbed(1))
        elif len(nums) == 1:
            output = nums[0]
        else:
            output = 0
        return output
                
        
        
    
t1 = [1,2,3,1]
print(Solution().rob(t1))

t2 = [2,7,9,3,1]
print(Solution().rob(t2))