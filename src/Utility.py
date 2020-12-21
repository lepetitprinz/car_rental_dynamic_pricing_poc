import os
import datetime as dt
from calendar import monthrange
from datetime import timedelta

import numpy as np
import pandas as pd


class Utility(object):
    # Path
    PATH_INPUT = os.path.join('..', 'input')
    PATH_MODEL = os.path.join('..', 'result', 'model')

    # Initial Setting
    TEST_SIZE = 0.2
    RANDOM_STATE = 2020
    AVG_UNAVAIL_CAPA = 2    # Average unavailable capacity

    # Data
    TYPE_GROUP: list = ['av', 'k3', 'su', 'vl']
    TYPE_MODEL: list = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
    TYPE_DATA: list = ['cnt', 'disc', 'util']
    TYPE_DATA_MAP: dict = {'cnt': 'cnt_cum', 'disc': 'cnt_cum', 'util': 'util_cum'}

    GRADE_1_6: list = ['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)',
                       'ALL NEW K3 (G)', '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
    MODEL_NAME_MAP: dict = {'아반떼 AD (G) F/L': 'av_ad', '올 뉴 아반떼 (G)': 'av_new',
                            'ALL NEW K3 (G)': 'k3', '쏘울 부스터 (G)': 'soul', '더 올 뉴 벨로스터 (G)': 'vlst'}
    MODEL_NAME_MAP_REV: dict = {'av_ad': '아반떼 AD (G) F/L', 'av_new': '올 뉴 아반떼 (G)',
                                'k3': 'ALL NEW K3 (G)', 'soul': '쏘울 부스터 (G)', 'vlst': '더 올 뉴 벨로스터 (G)'}

    # Rename
    RENAME_COL_RES: dict = {
            '예약경로': 'res_route', '예약경로명': 'res_route_nm', '계약번호': 'res_num',
            '고객구분': 'cust_kind', '고객구분명': 'cust_kind_nm', '총 청구액(VAT포함)': 'tot_fee',
            '예약모델': 'res_model', '예약모델명': 'res_model_nm', '차급': 'car_grd',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '대여기간(일)': 'rent_period_day', '대여기간(시간)': 'rent_period_time',
            'CDW요금': 'cdw_fee', '할인유형': 'discount_type', '할인유형명': 'discount_type_nm',
            '적용할인명': 'applied_discount', '적용할인율(%)': 'discount', '회원등급': 'member_grd',
            '구매목적': 'sale_purpose', '생성일': 'res_day', '차종': 'car_kind'}

    #####################
    # Load datasets
    #####################
    @staticmethod
    def load_res(time: str, status_update_day=''):
        """
        :param time: hx / re (history or recent)
        :param status_update_day: Update day of reservation status
        :return:
        """
        res = pd.DataFrame()
        if time == 're':
            data_path = os.path.join('..', 'input', 'res_status', ''.join(['res_status_', status_update_day, '.csv']))
            res = pd.read_csv(data_path, delimiter='\t',
                              dtype={'예약경로': int, '예약경로명': str, '계약번호': int, '고객구분': int, '고객구분명': str,
                                     '총 청구액(VAT포함)': str, '예약모델': str, '예약모델명': str, '차급': str,
                                     '대여일': str, '대여시간': str, '반납일': str, '반납시간': str, '대여기간(일)': int,
                                     '대여기간(시간)': int, 'CDW요금': str, '할인유형': str, '할인유형명': str,
                                     '적용할인명': str, '회원등급': str, '구매목적': str, '생성일': str, '차종': str})
        elif time == 'hx':
            data_path = os.path.join('..', 'input', 'res_status', 'res_hx.csv')
            res = pd.read_csv(data_path,
                              dtype={'res_num': str, 'res_route_nm': str, 'res_model_nm': str, 'rent_day': str,
                                     'rent_time': str, 'return_day': str, 'return_time': str, 'car_rent_fee': int,
                                     'cdw_fee': int, 'tot_fee': int, 'discount': float, 'res_day': str,
                                     'seasonality': int})

        return res

    @staticmethod
    def load_season(time: str):
        """
        :param time: hx / re (history or recent)
        :return:
        """
        # Load seasonal dataset
        data_path = os.path.join('..', 'input', 'seasonality', ''.join(['seasonality_', time, '.csv']))
        season = pd.read_csv(data_path, delimiter='\t', dtype={'date': str, 'seasonality': int})
        season = season.rename(columns={'date': 'rent_day'})
        season['rent_day'] = pd.to_datetime(season['rent_day'], format='%Y-%m-%d')

        return season

    @staticmethod
    def load_capacity(time: str, type_apply: str, unavail=False):
        """
        :param time: hx / re (history or Recent)
        :param type_apply: model / car
        :param unavail: unavailable capacity or not
        :return:
        """
        if not unavail:
            data_path = os.path.join('..', 'input', 'capa', ''.join(['capa_', time, '_', type_apply, '.csv']))
        else:
            data_path = os.path.join('..', 'input', 'capa', ''.join(['capa_', time, '_', type_apply, '_unavail.csv']))
        capacity = pd.read_csv(data_path, delimiter='\t', dtype={'date': str, 'model': str, 'capa': int})

        return capacity

    #####################
    # Cluster car models
    #####################
    @staticmethod
    def cluster_model(df: pd.DataFrame, type_apply: str):
        if type_apply == 'car':
            conditions = [
                df['res_model_nm'].isin(['아반떼 AD (G)', '아반떼 AD (G) F/L']),
                df['res_model_nm'] == '올 뉴 아반떼 (G)',
                df['res_model_nm'] == 'ALL NEW K3 (G)',
                df['res_model_nm'].isin(['쏘울 (G)', '쏘울 부스터 (G)']),
                df['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
            values = Utility.TYPE_MODEL
            df['res_model'] = np.select(conditions, values)

        elif type_apply == 'model':
            av = ['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)']
            k3 = ['ALL NEW K3 (G)']
            vl = ['더 올 뉴 벨로스터 (G)']
            su = ['쏘울 (G)', '쏘울 부스터 (G)']

            conditions = [
                df['res_model_nm'].isin(av),
                df['res_model_nm'].isin(k3),
                df['res_model_nm'].isin(vl),
                df['res_model_nm'].isin(su)]
            values = ['AVANTE', 'K3', 'VELOSTER', 'SOUL']
            df['res_model'] = np.select(conditions, values)

        return df

    @staticmethod
    def filter_model_grade(df: pd.DataFrame):
        df = df[df['res_model_nm'].isin(Utility.GRADE_1_6)]
        df = df.reset_index(drop=True)

        return df

    #######################
    # Methods of Capacity
    #######################
    @staticmethod
    def conv_mon_to_day(df: pd.DataFrame):
        months = np.sort(df['date'].unique())
        months.sort()
        fst_month = dt.datetime.strptime(months[0], '%Y%m')
        last_month = dt.datetime.strptime(months[-1], '%Y%m')
        last_day = last_month.replace(day=monthrange(last_month.year, last_month.month)[1])
        days = pd.date_range(start=fst_month, end=last_day)
        model_unique = df[['model', 'capa']].drop_duplicates()

        df_days = pd.DataFrame()
        for model, capa in zip(model_unique['model'], model_unique['capa']):
            temp = pd.DataFrame({'date': days, 'model': model, 'capa': capa})
            df_days = pd.concat([df_days, temp], axis=0)

        return df_days

    @staticmethod
    def apply_unavail_capa(capacity: pd.DataFrame, capa_unavail: pd.DataFrame):
        capa_unavail['date'] = pd.to_datetime(capa_unavail['date'], format='%Y%m%d')
        capa_new = pd.merge(capacity, capa_unavail, how='left', on=['date', 'model'],
                            left_index=True, right_index=False)
        capa_new = capa_new.fillna(0)
        capa_new['capa'] = capa_new['capa'] - capa_new['unavail']

        return capa_new

    @staticmethod
    def make_capa_map(df: pd.DataFrame):
        return {(date, Utility.MODEL_NAME_MAP[model]): capa for date, model, capa in zip(df['date'],
                                                                                         df['model'],
                                                                                         df['capa'])}

    # Lead time
    @staticmethod
    def get_lead_time():
        # Lead Time Setting
        lt = np.arange(-89, 1, 1)
        lt_vec = np.arange(-36, 1, 1)
        lt_to_lt_vec = {-1 * i: (((i // 7) + 24) * -1 if i > 28 else i * -1) for i in range(0, 90, 1)}

        return lt, lt_vec, lt_to_lt_vec

    @staticmethod
    def add_col_lead_time(df: pd.DataFrame):
        df['lead_time'] = df['rent_day'] - df['res_day']
        df['lead_time'] = df['lead_time'].to_numpy().astype('timedelta64[D]') / np.timedelta64(1, 'D')
        df['lead_time'] = df['lead_time'] * -1

        return df

    @staticmethod
    def add_col_lead_time_vec(df: pd.DataFrame):
        # Add lead time
        df['lead_time'] = df['rent_day'] - df['res_day']
        df['lead_time'] = df['lead_time'].to_numpy().astype('timedelta64[D]') / np.timedelta64(1, 'D')

        # Add lead time vector
        df['lead_time_vec'] = (df['lead_time'] // 7) + 24       # Lead time > 28 : 1 week
        df['lead_time_vec'] = np.where(df['lead_time_vec'] < 28,  # Lead time 1 ~ 27
                                       df['lead_time'].values,
                                       df['lead_time_vec'].values)
        df['lead_time_vec'] = np.where(df['lead_time_vec'] > 36, 36,  # Lead time > 36
                                       df['lead_time_vec'].values)

        # Change data type
        df['lead_time_vec'] = df['lead_time_vec'] * -1
        df['lead_time_vec'] = df['lead_time_vec'].astype(int)
        df = df.drop(columns=['lead_time'])

        return df

    @staticmethod
    def get_res_util(df: pd.DataFrame):
        res_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                df['rent_day'], df['rent_time'], df['return_day'], df['return_time'],
                df['res_day'], df['discount'], df['res_model']):

            day_hour = timedelta(hours=24)
            six_hour = timedelta(hours=6)
            date_range = pd.date_range(start=rent_d, end=return_d)  # days of rent periods
            date_len = len(date_range)
            fst = list(map(int, rent_t.split(':')))
            lst = list(map(int, return_t.split(':')))
            ft = timedelta(hours=fst[0], minutes=fst[1])  # time of rent day
            lt = timedelta(hours=lst[0] + 2, minutes=lst[1])  # time of return day

            f_util = 1
            l_util = 1
            # Classify reservation periods
            # If periods is more than 6 hours, utilization is 1
            if (day_hour - ft) < six_hour:
                f_util = (day_hour - ft) / six_hour
            # If periods is less than 6 hours, utilization is
            if lt < six_hour:
                l_util = lt / six_hour

            if date_len > 2:
                util = np.array(f_util)
                util = np.append(util, np.ones(date_len - 2))
                util = np.append(util, l_util)

            elif date_len == 2:
                util = np.array([f_util, l_util])

            else:
                util = 1
                if (lt - ft) < six_hour:
                    util = (lt - ft) / six_hour
                util = np.array([util])

            res_util.extend(np.array([
                date_range, [res_day] * date_len, util, [discount] * date_len, [model] * date_len]).T)
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util', 'discount', 'res_model'])

        return res_util_df

    @staticmethod
    def _get_res_util_bak(df: pd.DataFrame):
        res_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                df['rent_day'], df['rent_time'], df['return_day'], df['return_time'],
                df['res_day'], df['discount'], df['res_model']):

            day_hour = timedelta(hours=24)
            date_range = pd.date_range(start=rent_d, end=return_d)  # days of rent periods
            date_len = len(date_range)
            fst = list(map(int, rent_t.split(':')))
            lst = list(map(int, return_t.split(':')))
            ft = timedelta(hours=fst[0], minutes=fst[1])  # time of rent day
            lt = timedelta(hours=lst[0] + 2, minutes=lst[1])  # time of return day

            # Classify reservation periods
            f_util = (day_hour - ft) / day_hour
            l_util = lt / day_hour

            if date_len > 2:
                util = np.array(f_util)
                util = np.append(util, np.ones(date_len - 2))
                util = np.append(util, l_util)

            elif date_len == 2:
                util = np.array([f_util, l_util])

            else:
                util = (lt - ft) / day_hour
                util = np.array([util])

            res_util.extend(np.array([
                date_range,
                [res_day] * date_len,
                util,
                [discount] * date_len,
                [model] * date_len
            ]).T)
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util_rate', 'discount', 'res_model'])

        return res_util_df