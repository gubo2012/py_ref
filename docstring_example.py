# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:42:43 2019

@author: dxuser22
"""

def func1():
    """ func1: triple-doube quote """
    print('python doc example - doc - triple-doube quote')


def func2():
#    normal comments
    print('python doc example - normal comments')
    


if __name__ == '__main__': 
    """ main: triple-doube quote """

    func1()
    print(func1.__doc__)
    
    func2()
    print(func2.__doc__)


#help(func1)
#Help on function func1 in module __main__:
#
#func1()
#    func1: triple-doube quote
#
#
#help(func2)
#Help on function func2 in module __main__:
#
#func2()