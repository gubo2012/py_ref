# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:58:32 2019

@author: GuBo
"""

import pandas as pd
import sys

#UDF
import stock_io

if __name__ == '__main__':

    if len(sys.argv)>1:
        ticker = sys.argv[1]
    else:
        ticker = 'QQQ'

    df = pd.read_csv(stock_io.raw_data.format(ticker))

    df1 = df.copy()
    df1 = df1.rename(columns = {'Close':'close', 'High':'high', 'Low':'low'})

    df1.to_csv(stock_io.format_data.format(ticker), index=False)