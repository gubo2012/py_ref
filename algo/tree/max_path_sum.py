# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:44:35 2019

@author: GuBo
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    
    max_sum:int
    
    def trav_node(self, node):
        if not node:
            return 0
        else:
            l = self.trav_node(node.left)
            r = self.trav_node(node.right)
            node_max = max(l, 0) + max(r, 0) + node.val
            self.max_sum = max(self.max_sum, node_max)
            return node_max
    
    def maxPathSum(self, root: TreeNode) -> int:
        if not root:
            return 0
        else:
            self.max_sum = root.val
            self.trav_node(root)
            
        return self.max_sum
        
        
        
        
#t = [-10,9,20,None,None,15,7]   
t = [5,4,8,11,None,13,4,7,2,None,None,None,1]
dict_node = {}
i = 0
root = TreeNode(t[i])
dict_node[i] = root

while i < len(t):
    if t[i] != None:
        node = dict_node[i]
        
        l = i * 2 + 1
        r = i * 2 + 2
        
        if l < len(t):
            if t[l] != None:
                dict_node[l] = TreeNode(t[l]) 
                node.left = dict_node[l]
            else:
                node.left = None
        
        if r < len(t):
            if t[r] != None:
                dict_node[r] = TreeNode(t[r]) 
                node.right = dict_node[r]
            else:
                node.righ = None
    i += 1

print(Solution().maxPathSum(root))