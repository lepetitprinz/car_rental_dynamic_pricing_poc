# Import class
# Data Preprocessing
from DataPreprocessing import DataPrep

# Model
from TimeSeries import TimeSeries           # Model 1
from ResPredict import ResPredict           # Model 2
from DiscRecommend import DiscRecommend     # Model 3

# Data Post Processing
from WeeklyReport import WeeklyReport

# Sales Prediction model
from SalesPredict import SalesPredict

# Import libraries
import warnings
import numpy as np
import pandas as pd
from itertools import permutations
warnings.filterwarnings('ignore')


def data_preprocessing(res_status_ud_day: str, end_date: str):
    data_prep = DataPrep(end_date=end_date)
    # History dataset
    # data_prep.prep_res_hx()
    # Recent  dataset
    data_prep.prep_res_recent(res_status_ud_day=res_status_ud_day)

# Model 1 : Jeju visitor prediction
def model_1(start_date: str, end_date: str, n_test: int, pred_step: int):
    test_models = ['ar', 'arima', 'hw']  # AR / / ARIMA / Holt-winters
    # Parameters Grids
    param_grids = {
        'ar': {
            'lags': list(np.arange(1, 15, 1)),
            'trend': ['c', 't', 'ct']},
        'arima': {
            'order': list(permutations(np.arange(1, 8, 1), 3)),
            'trend': ['c', 't', 'ct']},
        'hw': {
            'trend': ['add', 'additive'],
            'damped_trend': [True, False]}}

    model = TimeSeries(start_date=start_date, end_date=end_date)

    # Train
    model.train(n_test=n_test, test_models=test_models, param_grids=param_grids)

    # Prediction
    model.predict(pred_step=pred_step)    # 12 weeks + 1 month


# Model 2
def model_2(start_date: str, end_date: str, apply_day: str,
            res_status_ud_day: str, disc_confirm_last_week: str, model_detail: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model = ResPredict(res_status_ud_day=res_status_ud_day)

    # Train
    # model.train(model_detail=model_detail)

    # Prediction
    model.predict(pred_days=pred_days, apply_day=apply_day,
                  disc_confirm_last_week=disc_confirm_last_week, model_detail=model_detail)


# Model 3
def model_3(start_date: str, end_date: str, apply_day: str, res_update_day: str,
            disc_confirm_last_week: str, model_detail: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model = DiscRecommend(res_update_day=res_update_day, apply_day=apply_day, model_detail=model_detail,
                          disc_confirm_last_week=disc_confirm_last_week)
    model.rec(pred_days=pred_days)

def model_sales_pred(start_date: str, end_date: str, apply_day: str,
                     res_status_ud_day: str, disc_confirm_last_week: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    # Predict reservation changing trend
    pred_sales = SalesPredict(res_status_ud_day=res_status_ud_day,
                              disc_confirm_last_week=disc_confirm_last_week)
    # pred_sales.data_preprocessing()
    # pred_sales.train()
    pred_sales.predict(pred_days=pred_days, apply_day=apply_day)

    # Recommend discount rate
    # disc_rec_lead_time = DiscRecLeadTime(res_update_day=res_update_day)
    # disc_rec_lead_time.rec(pred_days=pred_days, apply_day=apply_day)


def weekly_report(res_status_last_week: str, res_status_this_week: str, res_status_cancel_this_week: str,
                  res_confirm_last_week: str, disc_confirm_last_week: str, disc_rec_last_week: str,
                  start_day: str, end_day: str, apply_last_week: str, apply_this_week: str,
                  res_confirm_day_from: str, res_confirm_day_to: str):
    post_proc = WeeklyReport(res_status_last_week=res_status_last_week,
                             res_status_this_week=res_status_this_week,
                             res_status_cancel_this_week=res_status_cancel_this_week,
                             res_confirm_last_week=res_confirm_last_week,
                             disc_confirm_last_week=disc_confirm_last_week,
                             disc_rec_last_week=disc_rec_last_week,
                             start_day=start_day, end_day=end_day,
                             apply_last_week=apply_last_week,
                             apply_this_week=apply_this_week)

    # post_proc.post_process()
    post_proc.calc_sales(res_confirm_day_from=res_confirm_day_from,
                         res_confirm_day_to=res_confirm_day_to,
                         apply_last_week=apply_last_week)

##########################################
# Moin
##########################################
def main():
    # Recent reservation status (최신 예약현황)
    res_status_ud_day = '201210'        # 최신 예약건수 업데이트 날짜

    # Hyper-parameters for time series
    m1_start_date = '20191201'
    m1_end_date = '20201130'
    n_test = 84  # 12 weeks

    # Recommend hyper-parameter
    start_date = '2020/12/14'           # 할인율 산출 기간: 시작
    end_date = '2021/02/28'             # 할인율 산출 기간: 끝
    apply_day = '2020/12/14'            # Discount apply day (할인율 적용일)
    disc_confirm_last_week = '201204'        # Discount dataset on previous week (이전 확정 할인율)

    # Weekly Report hyper-paramter
    res_status_last_week = '201126'     #
    res_status_this_week = '201204'     #
    res_status_cancel_this_week = '201203'    # 취소 데이터 업데이트 날짜
    res_confirm_last_week = '201204'         # Reservation dataset on previous week (이전 확정 예약실적)
    disc_rec_last_week = '20201201'     #
    start_day = '20201201'
    end_day = '20210228'
    apply_last_week = '201127'      # Discount apply day on last week
    apply_this_week = '201204'      # Discount apply day on this week
    res_confirm_day_from = '201201'
    res_confirm_day_to = '201204'

    # Data Preprocessing
    data_preprocessing(res_status_ud_day=res_status_ud_day,end_date=end_date)

    # Model 1
    # model_1(start_date=m1_start_date,
    #         end_date=m1_end_date,
    #         n_test=n_test,
    #         pred_step=31+28+31)

    # Model 2
    model_2(start_date=start_date,
            end_date=end_date,
            apply_day=apply_day,
            res_status_ud_day=res_status_ud_day,
            disc_confirm_last_week=disc_confirm_last_week,
            model_detail='car')

    # Model 3
    model_3(start_date=start_date,
            end_date=end_date,
            apply_day=apply_day,
            res_update_day=res_status_ud_day,
            disc_confirm_last_week=disc_confirm_last_week,
            model_detail='car')

    # Sales Prediction
    # model_sales_pred(start_date=start_date,
    #                  end_date=end_date,
    #                  apply_day=apply_day,
    #                  res_update_day=res_status_ud_day,
    #                  disc_confirm_last_week=disc_confirm_last_week)

    # Data Post Processing
    # weekly_report(res_status_last_week=res_status_last_week,
    #               res_status_this_week=res_status_this_week,
    #               res_status_cancel_this_week=res_status_cancel_this_week,
    #               res_confirm_last_week=res_confirm_last_week,
    #               disc_confirm_last_week=disc_confirm_last_week,
    #               disc_rec_last_week=disc_rec_last_week,
    #               start_day=start_day, end_day=end_day,
    #               apply_last_week=apply_last_week,
    #               apply_this_week=apply_this_week,
    #               res_confirm_day_from=res_confirm_day_from,
    #               res_confirm_day_to=res_confirm_day_to)


# Run main function
main()
