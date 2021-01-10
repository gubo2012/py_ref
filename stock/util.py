import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import json


def mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return 100 * np.mean(np.abs((actual - prediction))/actual)


def get_sim_accuracy(data, final_prediction):
    
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
    return accuracy_1, accuracy_2


def print_results(accuracy_act, accuracy_pred,
                  mse, rmse, mape):
        
#    print('\n' + ma)
    print('Prediction vs Close:\t\t' + str(round(100*accuracy_act, 2))
          + '% Accuracy')
    print('Prediction vs Prediction:\t' + str(round(100*accuracy_pred, 2))
          + '% Accuracy')
    print('MSE:\t', mse,
          '\nRMSE:\t', rmse,
          '\nMAPE:\t', mape)

    
def show_cols(df, str_in):
    col_list = [col for col in df.columns if str_in in col]
    return col_list