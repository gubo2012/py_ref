# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:42:43 2019

@author: dxuser22
"""

import numpy.testing as npt
import math

# not auto run in pytest
def assert1():
    print('run test func assert1')
    assert 1==1
    assert math.pi == math.sqrt(math.pi * math.pi)
    

# auto run in pytest
def test_assert1():
    print('run test func test_assert1')
    assert 1==1
    assert math.pi == math.sqrt(math.pi * math.pi)


# auto run in pytest
def testassert1():
    print('run test func testassert1')
    assert 1==1
    assert math.pi == math.sqrt(math.pi * math.pi)


if __name__ == '__main__': 
    assert1()
    test_assert1()
    testassert1()