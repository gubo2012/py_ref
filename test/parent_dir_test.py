# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:12:54 2019

@author: dxuser22
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# or use os.chdir('..')

import docstring_example

def test_parent_dir():
    docstring_example.func1()

if __name__ == '__main__': 
    docstring_example.func1()
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))