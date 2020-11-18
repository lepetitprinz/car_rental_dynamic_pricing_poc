from Model1 import MODEL1
from Model2 import MODEL2
from Model3 import MODEL3

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# Model 1 : Jeju visitor prediction

def model_1():
    # Define hyper-parameters
    start_date = 20191101
    end_date = 20201031
    n_test = 28     # 4 weeks
    test_models = ['ar', 'hw']      # ar / Holt-winters

    # Parameters Grids
    param_grids = {
        'ar': {
            'lags': list(np.arange(1, 15, 1)),
            'trend': ['c', 't', 'ct']},
        'hw': {
            'trend': ['add', 'additive'],
            'damped_trend': [True, False]}}

    jeju_visitors = pd.read_csv(os.path.join('..', 'input', 'jeju_visit_daily.csv'), delimiter='\t')

    model_1 = MODEL1(visitor=jeju_visitors, start_date=20191101, end_date=20201031)

    # Train
    model_1.train(n_test=n_test, test_models=test_models, param_grids=param_grids)

    # Prediction
    model_1.predict(pred_step=88 + 31)

# Model 2
def model_2():
    # Define hyper-parameters

    model_2 = MODEL2()

    # model_2.train()

    # Prediction days
    start_date = '2020/12/21'
    end_date = '2021/02/15'
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model_2.predict(pred_days=pred_days)

def model_3():
    apply_day = '2020/11/23'
    curr_res_day = '201118'
    # Prediction days
    start_date = '2020/12/21'
    # end_date = '2021/02/15'
    end_date = '2021/02/15'
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model_3 = MODEL3(curr_res_day=curr_res_day)
    model_3.rec(pred_days=pred_days, apply_day=apply_day)

# Moin

# run model 1
# model_1()

# run model 2
# model_2()

# run model 3
model_3()