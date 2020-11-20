# Data Preprocessing
from DataPreprocessing import DataPrep

# Model
from Model1 import MODEL1
from Model2 import MODEL2
from Model3 import MODEL3

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# Model 1 : Jeju visitor prediction
def model_1(jeju_visitor: pd.DataFrame, start_date: int, end_date: int,
            n_test: int, test_models: list, param_grids: dict):

    model_1 = MODEL1(visitor=jeju_visitor, start_date=start_date, end_date=end_date)
    # Train
    model_1.train(n_test=n_test, test_models=test_models, param_grids=param_grids)
    # Prediction
    model_1.predict(pred_step=88 + 31)    # 12 weeks + 1 month

# Model 2
def model_2(pred_days: list, curr_res_day: str):
    model_2 = MODEL2(curr_res_day=curr_res_day)
    # Train
    # model_2.train()
    # Prediction
    model_2.predict(pred_days=pred_days)

def model_3(pred_days: list, apply_day: str, curr_res_day: str):
    model_3 = MODEL3(curr_res_day=curr_res_day)
    model_3.rec(pred_days=pred_days, apply_day=apply_day)

##################
# Moin
##################
# Data Preprocessing
# data_prep = DataPrep()
# data_prep.prep_res_hx()

# Model 1
jeju_visitors = pd.read_csv(os.path.join('..', 'input', 'jeju_visit_daily.csv'), delimiter='\t')

# Define hyper-parameters
m1_start_date = 20191101
m1_end_date = 20201031
n_test = 28  # 4 weeks
test_models = ['ar', 'hw']  # ar / Holt-winters

# Parameters Grids
param_grids = {
    'ar': {
        'lags': list(np.arange(1, 15, 1)),
        'trend': ['c', 't', 'ct']},
    'hw': {
        'trend': ['add', 'additive'],
        'damped_trend': [True, False]}}
# model_1(jeju_visitor=jeju_visitors, start_date=m1_start_date, end_date=m1_end_date,
#         n_test=n_test, test_models=test_models, param_grids=param_grids)

# Model 2
# Prediction days
start_date = '2020/12/21'
end_date = '2021/02/15'
curr_res_day = '201118'

pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')
model_2(pred_days=pred_days, curr_res_day=curr_res_day)

# Model 3
apply_day = '2020/11/23'
model_3(pred_days=pred_days, apply_day=apply_day, curr_res_day=curr_res_day)