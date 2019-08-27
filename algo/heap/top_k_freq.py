# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:43:41 2019

@author: GuBo
"""
import heapq

class Solution:
    def topKFrequent(self, nums, k: int):
        
        freqs = {}
        for i in nums:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1
        
        h = []
        for key, v in freqs.items():
            heapq.heappush(h, (v, key))
            
        outputs = heapq.nlargest(k, h)
        outputs = list(map(lambda x:x[1], outputs))
            
        return outputs
        
nums = [1,1,1,2,2,3]
k = 2
print(Solution().topKFrequent(nums, k))