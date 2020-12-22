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
import argparse
import warnings
import numpy as np
import pandas as pd
from itertools import permutations
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--dmd_pred', type=bool, default=False)
parser.add_argument('--data_prep', type=bool, default=False)
parser.add_argument('--res_pred', type=bool, default=False)
parser.add_argument('--disc_rec', type=bool, default=False)
parser.add_argument('--sales_pred', type=bool, default=False)
parser.add_argument('--weekly_report', type=bool, default=False)

args = parser.parse_args()


def data_preprocessing(res_status_ud_day: str, end_date: str, type_apply: str):
    data_prep = DataPrep(end_date=end_date, res_status_ud_day=res_status_ud_day)
    # History dataset
    data_prep.prep_res_history(type_apply=type_apply)
    # Recent  dataset
    data_prep.prep_res_recent(res_status_ud_day=res_status_ud_day,
                              type_apply=type_apply, time='re')


# Model 1 : Jeju visitor prediction
def dmd_predict(start_date: str, end_date: str, n_test: int, pred_step: int):
    # test_models = ['ar', 'arima', 'hw']  # AR / / ARIMA / Holt-winters\
    test_models = ['ar', 'hw']
    # Parameters Grids
    param_grids = {
        'ar': {
            'lags': list(np.arange(1, 15, 1)),
            'trend': ['c', 't', 'ct']},
        # 'arima': {
        #     'order': list(permutations(np.arange(1, 8, 1), 3)),
        #     'trend': ['c', 't', 'ct']},
        'hw': {
            'trend': ['add', 'additive'],
            'damped_trend': [True, False]}}

    model = TimeSeries(start_date=start_date, end_date=end_date)

    # Train
    model.train(n_test=n_test, test_models=test_models, param_grids=param_grids)

    # Prediction
    model.predict(pred_step=pred_step)    # 12 weeks + 1 month


# Model 2
def res_predict(start_date: str, end_date: str, apply_day: str,
                res_status_ud_day: str, disc_confirm_last_week: str, type_apply: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model = ResPredict(res_status_ud_day=res_status_ud_day, apply_day=apply_day)

    # Train
    model.train(type_apply=type_apply)

    # Prediction
    model.predict(pred_days=pred_days, disc_confirm_last_week=disc_confirm_last_week, type_apply=type_apply)


# Model 3
def disc_recommend(start_date: str, end_date: str, apply_day: str, res_update_day: str,
                   disc_confirm_last_week: str, type_apply: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    model = DiscRecommend(res_status_ud_day=res_update_day, apply_day=apply_day, type_apply=type_apply,
                          disc_confirm_last_week=disc_confirm_last_week)
    model.rec(pred_days=pred_days, type_apply=type_apply)


def sales_predict(start_date: str, end_date: str, apply_day: str,
                  res_status_ud_day: str, disc_confirm_last_week: str):
    pred_days = pd.date_range(start=start_date, end=end_date, freq='D')
    pred_days = pd.Series(pred_days).dt.strftime('%Y-%m-%d')

    # Predict reservation changing trend
    pred_sales = SalesPredict(res_status_ud_day=res_status_ud_day,
                              apply_day=apply_day,
                              disc_confirm_last_week=disc_confirm_last_week,
                              end_date=end_date)
    pred_sales.data_preprocessing(type_apply='car')
    pred_sales.train()
    pred_sales.predict(pred_days=pred_days)


def weekly_report(start_date_weekly: str, end_date_weekly: str,
                  apply_last_week: str, apply_this_week: str,
                  res_status_last_week: str, res_status_this_week: str, res_status_cancel_this_week: str,
                  res_confirm_last_week: str, disc_confirm_last_week: str, disc_rec_days: list,
                  res_confirm_day_from: str, res_confirm_day_to: str, res_confirm_bf: str):
    post_proc = WeeklyReport(res_status_last_week=res_status_last_week,
                             res_status_this_week=res_status_this_week,
                             res_status_cancel_this_week=res_status_cancel_this_week,
                             res_confirm_last_week=res_confirm_last_week,
                             disc_confirm_last_week=disc_confirm_last_week,
                             disc_rec_days=disc_rec_days,
                             start_day=start_date_weekly, end_day=end_date_weekly,
                             apply_last_week=apply_last_week,
                             apply_this_week=apply_this_week,
                             res_confirm_day_from=res_confirm_day_from,
                             res_confirm_day_to=res_confirm_day_to,
                             res_confirm_bf=res_confirm_bf)
    # Post process
    post_proc.post_process(type_apply='car', time='re')
    # Calculate sales
    post_proc.calc_sales(res_confirm_day_from=res_confirm_day_from,
                         res_confirm_day_to=res_confirm_day_to,
                         apply_last_week=apply_last_week,
                         type_apply='car')

##########################################
# Moin
##########################################
def main(args):
    # Hyper-parameters for time series
    time_series_start_date = '20191201'
    times_series_end_date = '20201130'
    n_test = 84  # 12 weeks

    # Recent reservation status (최신 예약현황)
    res_status_ud_day = '201216'        # 최신 예약건수 업데이트 날짜

    # Recommend hyper-parameter
    start_date = '20201218'     # 할인율 산출 기간: 시작
    end_date = '20210228'       # 할인율 산출 기간: 끝
    # start_date = '20201223'     # 할인율 산출 기간: 시작
    # end_date = '20210103'       # 할인율 산출 기간: 끝
    apply_day = '20201218'      # Recommend discount apply day (할인율 산출 및 적용일)
    disc_confirm_last_week = '201204'   # Discount dataset on previous week (이전 확정 할인율)

    # Weekly Report hyper-paramter
    start_date_weekly = '20201205'      # 보고서 산출 기간: 시작
    end_date_weekly = '20210228'        # 보고서 산출 기간: 끝
    apply_last_week = '201211'          # 지난주 할인율 산출 및 적용일
    apply_this_week = '201218'          # 이번주 할인율 산출 및 적용일
    res_status_last_week = '201210'     # 지난주 예약현황 업데이트 날짜
    res_status_this_week = '201216'     # 이번주 예약현황 업데이트 날짜
    res_status_cancel_this_week = '201216'  # 취소 데이터 업데이트 날짜
    res_confirm_bf = '201126'
    res_confirm_last_week = '201211'        # 지난주 예약실적 업데이트 날짜
    disc_rec_days = ['20201201', '20201207', '20201214']    # 할인율 추천 날짜 리스트
    res_confirm_day_from = '201205'         # 예약실적 확정일: 시작
    res_confirm_day_to = '201211'           # 예약실적 확정일: 끝

    # 1.Demand Prediction
    if args.dmd_pred:
        dmd_predict(start_date=time_series_start_date,
                    end_date=times_series_end_date,
                    n_test=n_test,
                    pred_step=31 + 28 + 31)

    # 2.Data Preprocessing
    if args.data_prep:
        data_preprocessing(res_status_ud_day=res_status_ud_day, end_date=end_date, type_apply='car')

    # 3.Reservation Prediction
    if args.res_pred:
        res_predict(start_date=start_date,
                    end_date=end_date,
                    apply_day=apply_day,
                    res_status_ud_day=res_status_ud_day,
                    disc_confirm_last_week=disc_confirm_last_week,
                    type_apply='car')

    # 4.Discount Recommendation
    if args.disc_rec:
        disc_recommend(start_date=start_date,
                       end_date=end_date,
                       apply_day=apply_day,
                       res_update_day=res_status_ud_day,
                       disc_confirm_last_week=disc_confirm_last_week,
                       type_apply='car')

    # 5.Sales Prediction
    if args.sales_pred:
        sales_predict(start_date=start_date,
                      end_date=end_date,
                      apply_day=apply_day,
                      res_status_ud_day=res_status_ud_day,
                      disc_confirm_last_week=disc_confirm_last_week)

    # 6.Weekly Report
    if args.weekly_report:
        weekly_report(start_date_weekly=start_date_weekly, end_date_weekly=end_date_weekly,
                      apply_last_week=apply_last_week, apply_this_week=apply_this_week,
                      res_status_last_week=res_status_last_week, res_status_this_week=res_status_this_week,
                      res_status_cancel_this_week=res_status_cancel_this_week,
                      res_confirm_last_week=res_confirm_last_week,
                      disc_confirm_last_week=disc_confirm_last_week,
                      disc_rec_days=disc_rec_days,
                      res_confirm_day_from=res_confirm_day_from,
                      res_confirm_day_to=res_confirm_day_to,
                      res_confirm_bf=res_confirm_bf)

# Run main function
main(args)
