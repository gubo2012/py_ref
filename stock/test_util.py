# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:56:22 2021

@author: GuBo
"""

import pandas as pd
import numpy as np

import util

def assert_no_prior_days_data(df, n_day):
    for i in range(n_day - 1):
        n = i+1 # {n}d
        cols = util.show_cols(df, 'lag{}d'.format(n))
        assert(len(cols)==0)