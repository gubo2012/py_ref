# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 22:31:59 2019

@author: GuBo
"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) <=1:
            return len(s)
        else:
            c_dict = {} # record when a char is last seen
            max_len = 0
            left = 0
            right = 0
            while right < len(s):
                if s[right] in c_dict: # already seen
                    left = max(left, c_dict[s[right]] + 1)
                
                c_dict[s[right]] = right
                max_len = max(max_len, right - left + 1)
                right +=1
            return max_len
        
        
        
#t = "abcabcbb"
#t = "pwwkew"
t = "abba"
print(Solution().lengthOfLongestSubstring(t))