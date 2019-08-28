# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 21:09:02 2019

@author: GuBo
"""

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self, root: TreeNode):
        from collections import defaultdict
        stack = defaultdict()
        
        def push_dict(node, level):
            stack.setdefault(level, []).append(node.val)
            if node.left != None:
                push_dict(node.left, level+1)
            if node.right != None:
                push_dict(node.right, level+1)
        
        output = []
        if root != None:
            push_dict(root, 0)
            for k, v in stack.items():
                output.append(v)
        
        return output
        
        
        
t = [3,9,20, None, None,15,7]
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

print(Solution().levelOrder(root))
    