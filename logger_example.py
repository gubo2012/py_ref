# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:56:21 2019

@author: dxuser22
"""

from logger import my_logger, my_timer

@my_logger
@my_timer
def func1():
    print('start of func1')
    for i in range(10):
        print(i * i)
    print('end of func1')

if __name__ == '__main__': 
    func1()