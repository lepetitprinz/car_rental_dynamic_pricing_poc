# Import class
# Data Preprocessing
from DataPreprocessing import DataPrep

# Model
from TimeSeries import TimeSeries           # Model 1
from ResPredict import ResPredict           # Model 2
from DiscRecommend import DiscRecommend     # Model 3

# Data Post Processing
from DataPostProcessing import DataPostProcessing

# Sales Prediction model
from SalesPredict import SalesPredict

# Import libraries
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


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
def model_1(start_date: int, end_date: int, n_test: int):
    test_models = ['ar', 'hw']  # ar / Holt-winters
    # Parameters Grids
    param_grids = {
        'ar': {
            'lags': list(np.arange(1, 15, 1)),
            'trend': ['c', 't', 'ct']},
        'hw': {
            'trend': ['add', 'additive'],
            'damped_trend': [True, False]}}

    model = TimeSeries(start_date=start_date, end_date=end_date)

    # Train
    model.train(n_test=n_test, test_models=test_models, param_grids=param_grids)

    # Prediction
    model.predict(pred_step=88 + 31)    # 12 weeks + 1 month


# Model 2
def model_2(start_date: str, end_date: str, res_update_day: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model = ResPredict(res_update_day=res_update_day)

    # Train
    model.train(model_detail='model')

    # Prediction
    model.predict(pred_days=pred_days, model_detail='model')


# Model 3
def model_3(start_date: str, end_date: str, apply_day: str, res_update_day: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model = DiscRecommend(res_update_day=res_update_day)
    model.rec(pred_days=pred_days, apply_day=apply_day)


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


def data_post_processing(update_day_before: str, update_day_after: str, update_day_cancel: str,
                         res_complete_day: str, disc_rec_day: str,
                         start_day: str, end_day: str):
    post_proc = DataPostProcessing(update_day_before=update_day_before,
                                   update_day_after=update_day_after,
                                   update_day_cancel=update_day_cancel,
                                   res_complete_day=res_complete_day,
                                   disc_rec_day=disc_rec_day,
                                   start_day=start_day,
                                   end_day=end_day)
    post_proc.post_process()

##########################################
# Moin
##########################################
def main():
    # Hyper-parameters
    # m1_start_date = 20191101
    # m1_end_date = 20201031
    # n_test = 28  # 4 weeks
    # Prediction days
    res_update_day = '201203'
    start_date = '2020/12/07'
    end_date = '2021/02/28'
    apply_day = '2020/12/07'

    # Data Post Processing
    update_day_before = '201126'    #
    update_day_after = '201204'     #
    update_day_cancel = '201203'    # 취소 데이터 업데이트 날짜
    res_complete_day = '201203'     # 실적
    disc_rec_day = '20201201'
    start_day = '20201201'
    end_day = '20210228'

    # Data Preprocessing
    # data_preprocessing(update_day=res_update_day)

    # Model 1
    # model_1(start_date=m1_start_date,
    #         end_date=m1_end_date,
    #         n_test=n_test)

    # Model 2
    # model_2(start_date=start_date,
    #         end_date=end_date,
    #         res_update_day=res_update_day)

    # Model 3
    # model_3(start_date=start_date,
    #         end_date=end_date,
    #         apply_day=apply_day,
    #         res_update_day=res_update_day)

    # Sales Prediction
    model_sales_pred(start_date=start_date,
                     end_date=end_date,
                     apply_day=apply_day,
                     res_update_day=res_update_day)

    # Data Post Processing
    # data_post_processing(update_day_before=update_day_before,
    #                      update_day_after=update_day_after,
    #                      update_day_cancel=update_day_cancel,
    #                      res_complete_day=res_complete_day,
    #                      disc_rec_day=disc_rec_day,
    #                      start_day=start_day,
    #                      end_day=end_day)


# Run main function
main()
