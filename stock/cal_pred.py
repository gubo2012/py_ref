import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from pmdarima import auto_arima
import pmdarima as pm
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# from keras.callbacks import EarlyStopping
# from talib import abstract
# import json
import sys

# UDF 
from models import get_arima, get_lstm
from ta_util import get_functions
import stock_io



if __name__ == '__main__':

    if len(sys.argv)>1:
        ticker = sys.argv[1]
    else:
        ticker = 'QQQ'

    print('ticker:', ticker)

    data = pd.read_csv(stock_io.format_data.format(ticker), header=0).tail(1500).reset_index(drop=True)

    
    # Initialize moving averages from Ta-Lib, store functions in dictionary
    talib_moving_averages, functions = get_functions()

    # optimized_period = {'MIDPOINT': 59}
    optimized_period = {'MIDPOINT': 60}

    simulation = {}
    for ma in optimized_period:
        # Split data into low volatility and high volatility time series
        low_vol = functions[ma](data, optimized_period[ma])
        high_vol = data['close'] - low_vol

        # Generate ARIMA and LSTM predictions
        print('\nWorking on ' + ma + ' predictions')
        try:
            low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape = get_arima(low_vol, 1000, 252)
        except:
            print('ARIMA error, skipping to next MA type')
            continue

        df_low = pd.Series(low_vol_prediction) 
        df_low.to_csv(stock_io.file_pred_low.format(ticker), index=False)


        high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape = get_lstm(high_vol, 1000, 252)

        df_high = pd.Series(high_vol_prediction) 
        df_high.to_csv(stock_io.file_pred_high.format(ticker), index=False)        

