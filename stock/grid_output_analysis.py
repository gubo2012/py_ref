# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:47:33 2021

@author: GuBo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import stock_io
import ts_to_features
import ta_util
import util
import stock_ml
import stock_fe
import stock_bt

import json
import pickle


ticker = 'SPY'

pkl_file = open(stock_io.grid_search_output.format(ticker), 'rb')
grid_search_results = pickle.load(pkl_file)
pkl_file.close()