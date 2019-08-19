# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:53:31 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

def draw(outs, denom, succeed_flag=True):
    if succeed_flag:
        return outs/denom
    else:
        return (denom - outs)/denom


def draw_pf(outs, denom=47):
    draw_1 = outs/denom

    #draw_1_only_10 = draw_1 * (denom - 1 - (outs - 1))/(denom - 1)
    draw_1_only_10 = draw(outs, denom) * draw(outs-1, denom -1, False)
    
    #draw_2 = draw_1 * (outs - 1)/(denom -1)
    draw_2 = draw_1 * draw(outs-1, denom-1)
    
    #draw_1_only_01 = (denom - outs)/denom * (outs) / (denom -1)
    draw_1_only_01 = draw(outs, denom, False) * draw(outs, denom -1)
    assert draw_1_only_10 == draw_1_only_01    
    
    draw_1_only = draw_1_only_10 + draw_1_only_01
    return draw_1_only, draw_2


def draw_pt(outs, denom=46):
    return draw(outs, denom)


def break_even_prob(pot, r):
    return r/(pot + r * 2)

pot = 40

outs = 9

denom = 50 - 3

d1, d2 = draw_pf(outs, denom)

print(d1/outs, (d1+d2)/outs)

dt1 = draw_pt(outs)
print(dt1/outs)

r1 = pot / 2
pot2 = pot + r1 * 2

r2 = pot2 / 3

ev2 = (pot2 + r2) * dt1 + (- r2) * (1-dt1)

ev1 = (pot + r1) * draw(outs, denom) + (-r1) * (1-draw(outs, denom))