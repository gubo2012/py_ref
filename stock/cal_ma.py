import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from pmdarima import auto_arima
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from keras.callbacks import EarlyStopping
from talib import abstract
import json

import sys

# UDF
import stock_io
import ta_util


if __name__ == '__main__':

    threshold = 0.05

    if len(sys.argv)>1:
        ticker = sys.argv[1]
    else:
        ticker = 'QQQ'

    if len(sys.argv)>2:
        threshold_int = int(sys.argv[2])
        threshold = float(threshold_int) / 100

    print('ticker:', ticker)

    # Load historical data
    data = pd.read_csv(stock_io.format_data.format(ticker), header=0).tail(1500).reset_index(
        drop=True)


    talib_moving_averages, functions = ta_util.get_functions()


    # Determine kurtosis "K" values for MA period 4-99
    kurtosis_results = {'period': []}
    for i in range(4, 100):
        kurtosis_results['period'].append(i)
        for ma in talib_moving_averages:
            # Run moving average, remove last 252 days (used later for test data set), trim MA result to last 60 days
            ma_output = functions[ma](data[:-252], i).tail(60)

            # Determine kurtosis "K" value
            k = kurtosis(ma_output, fisher=False)

            # add to dictionary
            if ma not in kurtosis_results.keys():
                kurtosis_results[ma] = []
            kurtosis_results[ma].append(k)

    kurtosis_results = pd.DataFrame(kurtosis_results)
    kurtosis_results.to_csv('kurtosis_results.csv')

    # Determine period with K closest to 3 +/-5%
    optimized_period = {}
    for ma in talib_moving_averages:
        difference = np.abs(kurtosis_results[ma] - 3)
        df = pd.DataFrame({'difference': difference, 'period': kurtosis_results['period']})
        df = df.sort_values(by=['difference'], ascending=True).reset_index(drop=True)
        if df.at[0, 'difference'] < 3 * threshold:
            optimized_period[ma] = df.at[0, 'period']
        else:
            print(ma + ' is not viable, best K greater or less than 3 +/-{}%'.format(threshold * 100))

    print('\nOptimized periods:', optimized_period)
