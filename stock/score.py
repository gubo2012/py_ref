import pandas as pd
import numpy as np
#from scipy.stats import kurtosis
#from pmdarima import auto_arima
#import pmdarima as pm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
#from keras.callbacks import EarlyStopping
#from talib import abstract
import json


# UDF 
import util 
import stock_io



if __name__ == '__main__':

    ticker = 'QQQ'

    data = pd.read_csv(stock_io.format_data.format(ticker), header=0).tail(1500).reset_index(drop=True)
    
    low_vol_prediction = pd.read_csv(stock_io.file_pred_low.format(ticker), header=None)
    high_vol_prediction = pd.read_csv(stock_io.file_pred_high.format(ticker), header=None)
    
    final_prediction = pd.Series(low_vol_prediction[0]) + pd.Series(high_vol_prediction[0])

    mse = mean_squared_error(final_prediction.values, data['close'].tail(252).values)
    rmse = mse ** 0.5
    mape = util.mean_absolute_percentage_error(data['close'].tail(252).reset_index(drop=True), final_prediction)
    
    accuracy_act, accuracy_pred = util.get_sim_accuracy(data, final_prediction)
    
    util.print_results(accuracy_act, accuracy_pred,
                       mse, rmse, mape)
    
    


#
#        simulation[ma] = {'low_vol': {'prediction': low_vol_prediction, 'mse': low_vol_mse,
#                                      'rmse': low_vol_rmse, 'mape': low_vol_mape},
#                          'high_vol': {'prediction': high_vol_prediction, 'mse': high_vol_mse,
#                                       'rmse': high_vol_rmse},
#                          'final': {'prediction': final_prediction.values.tolist(), 'mse': mse,
#                                    'rmse': rmse, 'mape': mape},
#                          'accuracy': {'prediction vs close': accuracy_1, 'prediction vs prediction': accuracy_2}}
#
#        # save simulation data here as checkpoint
#        with open('simulation_data.json', 'w') as fp:
#            json.dump(simulation, fp)
#
#    for ma in simulation.keys():
#        print('\n' + ma)
#        print('Prediction vs Close:\t\t' + str(round(100*simulation[ma]['accuracy']['prediction vs close'], 2))
#              + '% Accuracy')
#        print('Prediction vs Prediction:\t' + str(round(100*simulation[ma]['accuracy']['prediction vs prediction'], 2))
#              + '% Accuracy')
#        print('MSE:\t', simulation[ma]['final']['mse'],
#              '\nRMSE:\t', simulation[ma]['final']['rmse'],
#              '\nMAPE:\t', simulation[ma]['final']['mape'])