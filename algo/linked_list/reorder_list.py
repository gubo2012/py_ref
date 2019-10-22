# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:59:53 2019

@author: GuBo
"""

# Definition for singly-linked list.
class ListNode:
    
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head != None:
            d = 0
            curr = head 
            stack = []
            while curr.next != None:
                d += 1
                curr = curr.next
                stack.append(curr)
            
            curr = head
            i = 0
            
            if d % 2 == 0:
                
                while i < d // 2:
                    tmp = stack.pop()
                    tmp.next = curr.next
                    curr.next = tmp
                    curr = tmp.next
                    i += 1
                curr.next = None
            
            else:
                
                while i < (d-1) // 2:
                    tmp = stack.pop()
                    tmp.next = curr.next
                    curr.next = tmp
                    curr = tmp.next
                    i += 1
                curr.next.next = None
            
            
        
            
        

l0 = ListNode(1)
l1 = ListNode(2)
l2 = ListNode(3)
l3 = ListNode(4)
l4 = ListNode(5)

l0.next = l1
l1.next = l2
l2.next = l3
l3.next = l4

def print_l(x):
    print(x.val)
    if x.next != None:
        print_l(x.next)
    else:
        print('end')
        
print_l(l0)
Solution().reorderList(l0)
print_l(l0)
    