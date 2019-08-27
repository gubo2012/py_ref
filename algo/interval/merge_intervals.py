# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:16:55 2019

@author: GuBo
"""

class Solution:
    def merge(self, intervals):
        
        if len(intervals) > 1:
                    
            intervals.sort(key = lambda x:x[0])
            
            i = len(intervals) - 1
            while i > 0: # backward once
                if intervals[i][0] <= intervals[i-1][1]:
                    intervals[i-1][1] = max(intervals[i-1][1], intervals[i][1])
                    del intervals[i]
                i -= 1
            
            if len(intervals) > 1:
                i = 0
                while i < len(intervals) - 1: # forward once
                    if intervals[i+1][0] <= intervals[i][1]:
                        intervals[i][1] = max(intervals[i][1], intervals[i+1][1])
                        del intervals[i+1]
                    else:
                        i += 1
            return intervals
            
        else:
            return intervals
        


inputs = [[1,3],[2,6],[8,10],[15,18]]
print(Solution().merge(inputs))