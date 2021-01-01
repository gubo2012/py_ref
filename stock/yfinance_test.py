# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:44:52 2021

@author: GuBo
"""

# run in yfin env
# https://pypi.org/project/yfinance/
# env not set up correctly; still not working


import yfinance as yf

msft = yf.Ticker("MSFT")

# get stock info
msft.info

# get historical market data
hist = msft.history(period="max")