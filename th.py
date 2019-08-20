# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:53:31 2019

@author: dxuser22
"""

import pandas as pd
import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt 
import seaborn as sns

rake_rate = 0.1
n = 1 # players against

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
#    print(outs)    
#    assert draw_1_only_10 == draw_1_only_01    
    npt.assert_almost_equal (draw_1_only_10, draw_1_only_01)
    
    draw_1_only = draw_1_only_10 + draw_1_only_01
    return draw_1_only, draw_2


def draw_pt(outs, denom=46):
    return draw(outs, denom)


def break_even_prob(pot, r,
                    n = 1, rake_rate=0):
    if rake_rate>0:
        tmp = r/((pot + r * n)*(1-rake_rate) + r )
    else:
        tmp = r/(pot + r * (n+1))
    return tmp


def format_outs(outs):
    d1, d2 = draw_pf(outs) 
    dpt = draw_pt(outs)
    return (outs, d1+d2, be_prob_to_r(d1+d2), dpt, be_prob_to_r(dpt))


def be_prob_to_r(prob):
    if prob < 0.5:
        tmp = prob/(1-2*prob)
    else:
        tmp = -1
    return tmp


def prob_to_commit(prob, n = 1):
    r = be_prob_to_r(prob)
    pot_new = 1 + r * (n+1)
    r_t = be_prob_to_r(prob) * pot_new
    return r + r_t


def call_ev(r, prob,
            n=1, rake_rate=0):
    ev = (1 + r*n) * (1-rake_rate) * prob - r * (1-prob)
    return ev   
    

def raise_ev(r, my_prob,
             n=1, rake_rate=0, fold_rate = 0.5):
    keep = 1 - rake_rate
    fold_pct = (1-my_prob) * fold_rate
    ev = 1 * keep * fold_pct + \
    (1 + r*n) * keep * r + \
    (-r) * (1 - my_prob - fold_pct)
    return ev


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


list_outs = list(range(5, 16))
list_pf = list(map(format_outs, list_outs))

df = pd.DataFrame(list_pf, columns=['o', 'pf', 'r_f', 'pt', 'r_t'])