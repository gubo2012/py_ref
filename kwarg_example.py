# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:35:20 2019

@author: dxuser22
"""
import inspect
import sys

# outside funcs example
from yield_example import nextSquare
import hanoi

def myFun(arg1, arg2, arg3): 
    print("arg1:", arg1) 
    print("arg2:", arg2) 
    print("arg3:", arg3) 
      
    
def myFun2(arg1, arg2): 
    print("arg1:", arg1) 
    print("arg2:", arg2) 


#def myFun3(arg1, arg2=2, *args3, **kwarg4): pass
def myFun3(arg1, arg2=2, arg2_2=2.2, *args3, **kwarg4): pass


# Now we can use *args or **kwargs to 
# pass arguments to this function :  
args = ("Geeks", "for", "Geeks") 
myFun(*args) 
  
kwargs = {"arg1" : "Geeks", "arg2" : "for", "arg3" : "Geeks"} 
myFun(**kwargs) 

# not working if pass more params then declare
#myFun2(*args) 
#myFun2(**kwargs) 

print(inspect.getfullargspec(myFun))

inspect_output = inspect.getfullargspec(myFun3)
print(inspect_output)
print(inspect_output[0], inspect_output[1])

# get all func names

hanoi.move(0, 2, 3)

print(sys.modules[__name__])
print(inspect.getmembers(sys.modules[__name__],
      predicate = lambda f: inspect.isfunction(f) and f.__module__ == __name__ ))

func_list = inspect.getmembers(sys.modules[__name__],
#      predicate = lambda f: inspect.isfunction(f) and f.__module__ == __name__ )
    predicate = lambda f: inspect.isfunction(f) )

func_name_list = list(map(lambda x:x[0], func_list))