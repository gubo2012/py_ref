# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:58:23 2019

@author: GuBo
"""

class Solution:
    def findMin(self, nums) -> int:
        i = 0
        j = len(nums)-1
        while i<j:
            if nums[j] < nums[i]:
                mid = int((i+j)/2)
                if nums[mid] < nums[i]:
                    j = mid
                else:
                    i = mid + 1
            else:
                break
        return nums[i]
                    

        
        
t1 = [3,4,5,1,2] 
print(Solution().findMin(t1))

t2 = [4,5,6,7,0,1,2]
print(Solution().findMin(t2))