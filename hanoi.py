# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:00:07 2019

@author: 80124664
"""


# tower 0, 1, 2
def move(t_from, t_to, move_n):
    t_other = 3 - t_from - t_to # 3 = 0 + 1 + 2
    if move_n == 1:
        print('move 1 from {} to {}'.format(t_from, t_to))
    else:
        move(t_from, t_other, move_n - 1)
        move(t_from, t_to, 1)
        move(t_other, t_to, move_n - 1)    

n = 4

move(0, 2, n)

