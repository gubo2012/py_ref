import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from pmdarima import auto_arima
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from talib import abstract
import json


# UDF 
from util import mean_absolute_percentage_error, get_arima, get_lstm
import stock_io



if __name__ == '__main__':

    ticker = 'SPY'

    data = pd.read_csv(stock_io.format_data.format(ticker), header=0).tail(1500).reset_index(drop=True)

    
    # Initialize moving averages from Ta-Lib, store functions in dictionary
    talib_moving_averages, functions = util.get_functions()

    optimized_period = {'MIDPOINT': 59}

    simulation = {}
    for ma in optimized_period:
        # Split data into low volatility and high volatility time series
        low_vol = functions[ma](data, optimized_period[ma])
        high_vol = data['close'] - low_vol

        # Generate ARIMA and LSTM predictions
        print('\nWorking on ' + ma + ' predictions')
        # try:
        #     low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape = get_arima(low_vol, 1000, 252)
        # except:
        #     print('ARIMA error, skipping to next MA type')
        #     continue

        high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape = get_lstm(high_vol, 1000, 252)

        # final_prediction = pd.Series(low_vol_prediction) + pd.Series(high_vol_prediction)
        final_prediction = pd.Series(high_vol_prediction) 
        final_prediction.to_csv('final_prediction_high.csv', index=False)

        mse = mean_squared_error(final_prediction.values, data['close'].tail(252).values)
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(data['close'].tail(252).reset_index(drop=True), final_prediction)

        # Generate prediction accuracy
        actual = data['close'].tail(252).values
        result_1 = []
        result_2 = []
        for i in range(1, len(final_prediction)):
            # Compare prediction to previous close price
            if final_prediction[i] > actual[i-1] and actual[i] > actual[i-1]:
                result_1.append(1)
            elif final_prediction[i] < actual[i-1] and actual[i] < actual[i-1]:
                result_1.append(1)
            else:
                result_1.append(0)

            # Compare prediction to previous prediction
            if final_prediction[i] > final_prediction[i-1] and actual[i] > actual[i-1]:
                result_2.append(1)
            elif final_prediction[i] < final_prediction[i-1] and actual[i] < actual[i-1]:
                result_2.append(1)
            else:
                result_2.append(0)

        accuracy_1 = np.mean(result_1)
        accuracy_2 = np.mean(result_2)

        simulation[ma] = {'low_vol': {'prediction': low_vol_prediction, 'mse': low_vol_mse,
                                      'rmse': low_vol_rmse, 'mape': low_vol_mape},
                          'high_vol': {'prediction': high_vol_prediction, 'mse': high_vol_mse,
                                       'rmse': high_vol_rmse},
                          'final': {'prediction': final_prediction.values.tolist(), 'mse': mse,
                                    'rmse': rmse, 'mape': mape},
                          'accuracy': {'prediction vs close': accuracy_1, 'prediction vs prediction': accuracy_2}}

        # save simulation data here as checkpoint
        with open('simulation_data.json', 'w') as fp:
            json.dump(simulation, fp)

    for ma in simulation.keys():
        print('\n' + ma)
        print('Prediction vs Close:\t\t' + str(round(100*simulation[ma]['accuracy']['prediction vs close'], 2))
              + '% Accuracy')
        print('Prediction vs Prediction:\t' + str(round(100*simulation[ma]['accuracy']['prediction vs prediction'], 2))
              + '% Accuracy')
        print('MSE:\t', simulation[ma]['final']['mse'],
              '\nRMSE:\t', simulation[ma]['final']['rmse'],
              '\nMAPE:\t', simulation[ma]['final']['mape'])