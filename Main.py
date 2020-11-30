# Data Preprocessing
from DataPreprocessing import DataPrep

# Model
from TimeSeries import TimeSeries           # Model 1
from ResPredict import ResPredict           # Model 2
from DiscRecommend import DiscRecommend     # Model 3

# Sales Prediction model
from SalesPredict import SalesPredict

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

def data_preprocessing(update_day: str):
    data_prep = DataPrep()
    # History dataset
    data_prep.prep_res_hx()
    # Recent  dataset
    data_prep.prep_res_recent(update_day=update_day)

    print('')
    print("Data preprocessing is finished.")
    print('')

# Model 1 : Jeju visitor prediction
def model_1(start_date: int, end_date: int,
            n_test: int):
    jeju_visitors = pd.read_csv(os.path.join('..', 'input', 'jeju_visit_daily.csv'), delimiter='\t')
    test_models = ['ar', 'hw']  # ar / Holt-winters
    # Parameters Grids
    param_grids = {
        'ar': {
            'lags': list(np.arange(1, 15, 1)),
            'trend': ['c', 't', 'ct']},
        'hw': {
            'trend': ['add', 'additive'],
            'damped_trend': [True, False]}}

    model_1 = TimeSeries(visitor=jeju_visitors, start_date=start_date, end_date=end_date)

    # Train
    model_1.train(n_test=n_test, test_models=test_models, param_grids=param_grids)
    # Prediction
    model_1.predict(pred_step=88 + 31)    # 12 weeks + 1 month

# Model 2
def model_2(start_date: str, end_date: str, res_update_day: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model_2 = ResPredict(res_update_day=res_update_day)

    # Train
    # model_2.train()

    # Prediction
    model_2.predict(pred_days=pred_days)

# Model 3
def model_3(start_date: str, end_date: str, apply_day: str, res_update_day: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model_3 = DiscRecommend(res_update_day=res_update_day)
    model_3.rec(pred_days=pred_days, apply_day=apply_day)

def model_sales_pred(start_date: str, end_date: str, apply_day: str, res_update_day: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    # Predict reservation changing trend
    pred_sales = SalesPredict(res_update_day=res_update_day)
    # pred_sales.data_preprocessing()
    # pred_sales.train()
    pred_sales.predict(pred_days=pred_days, apply_day=apply_day)

    # Recommend discount rate
    # disc_rec_lead_time = DiscRecLeadTime(res_update_day=res_update_day)
    # disc_rec_lead_time.rec(pred_days=pred_days, apply_day=apply_day)

##########################################
# Moin
##########################################
# Hyper-paramters
res_update_day = '201126'
m1_start_date = 20191101
m1_end_date = 20201031
n_test = 28  # 4 weeks
# Prediction days
# start_date = '2020/12/01'
# start_date = '2020/12/10'
start_date = '2021/02/26'
end_date = '2021/02/28'
apply_day = '2020/12/01'

# Data Preprocessing
# data_preprocessing(update_day=res_update_day)

# Model 1
# model_1(start_date=m1_start_date, end_date=m1_end_date, n_test=n_test,)

# Model 2
# model_2(start_date=start_date, end_date=end_date, res_update_day=res_update_day)

# Model 3
# model_3(start_date=start_date, end_date=end_date, apply_day=apply_day, res_update_day=res_update_day)

# Model lead time
model_sales_pred(start_date=start_date, end_date=end_date, apply_day=apply_day, res_update_day=res_update_day)