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
        # draw fail
        return (denom - outs)/denom


def draw_postflop(outs, denom=47):
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
    d1, d2 = draw_postflop(outs) 
    dpt = draw_pt(outs)
    return (outs, d1+d2, be_prob_to_call(d1+d2), prob_to_commit(d1+d2), dpt, be_prob_to_call(dpt))


# breakeven prob to call; return pct of pot, e.g., for 0.20 prob, can call up to 0.34 pot
def be_prob_to_call(prob):
    if prob < 0.5:
        tmp = prob/(1-2*prob)
    else:
        tmp = -1
    return tmp


# total commit @ turn, river
def prob_to_commit(prob, n = 1):
    r = be_prob_to_call(prob)
    pot_new = 1 + r * (n+1)
    r_t = be_prob_to_call(prob) * pot_new
    return r + r_t


# ev to call
def call_ev(r, prob,
            n=1, rake_rate=0):
    # r is the raise amount = to call amount
    ev = (1 + r*n) * (1-rake_rate) * prob - r * (1-prob)
    return ev   
    

def raise_ev(r, my_prob,
             n=1, rake_rate=0, fold_rate = 0.5):
    keep = 1 - rake_rate

    ev = 1 * keep * fold_rate + \
    (1 + r*n) * keep * (1-fold_rate) * my_prob + \
    (-r) * (1-fold_rate) * (1 - my_prob )
    return ev


if __name__ == '__main__':
    
    pot = 40
    
    outs = 9
    
    denom = 50 - 3
    
    d1, d2 = draw_postflop(outs, denom)
    
    print('{} outs at post flop, {:.3f} per out draw 1 only, {:.3f} per out draw back2back'.format(outs, d1/outs, (d1+d2)/outs))
    print('{:.3f} draw success 1 only, {:.3f} draw success back2back'.format(d1, (d1+d2)))
    
    dt1 = draw_pt(outs)
    print('{} outs at post turn, {:.3f} per out @ river, {:.3f} total @ river'.format(outs, dt1/outs, dt1))
    
    r1 = pot / 2
    pot2 = pot + r1 * 2
    
    r2 = pot2 / 3
    
    ev2 = (pot2 + r2) * dt1 + (- r2) * (1-dt1)
    
    ev1 = (pot + r1) * draw(outs, denom) + (-r1) * (1-draw(outs, denom))
    
    
    list_outs = list(range(5, 16))
    list_postflop = list(map(format_outs, list_outs))
    
    df = pd.DataFrame(list_postflop, columns=['outs', 'pf', 'call_up_to_pot', 'total_commit', 'pt', 'r_t'])