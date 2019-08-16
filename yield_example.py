# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:52:11 2019

@author: dxuser22
"""

# A Python program to generate squares from 1 
# to 100 using yield and therefore generator 
  
# An infinite generator function that prints 
# next square number. It starts with 1 
def nextSquare(): 
    i = 1; 
  
    # An Infinite loop to generate squares  
#    while True:
    while i < 100: 
        yield i*i                 
        i += 1  # Next execution resumes  
                # from this point      
  
# Driver code to test above generator  
# function 
for num in nextSquare(): 
    if num > 100: 
         break    
    print(num) 
    
aaa = list(nextSquare())