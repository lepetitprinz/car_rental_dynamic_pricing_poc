import os
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd


class DataPrep(object):

    def __init__(self):
        self.load_path = os.path.join('..', 'input')
        self.res_hx: pd.DataFrame = pd.DataFrame()
        self.res_curr: pd.DataFrame = pd.DataFrame()
        self.capa_curr: pd.DataFrame = pd.DataFrame()

        # car grade
        self.grade_1_6 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)', '아반떼 AD (G)', '아반떼 AD (G) F/L',
                          '올 뉴 아반떼 (G)', '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']

    def prep_res_hx(self):
        # Make reservation history dataset
        # self._make_res_hx()

        # Load reservation history
        res_hx = pd.read_csv(os.path.join(self.load_path, 'reservation', 'res_hx.csv'),
                             dtype={'res_num': str, 'res_route_nm': str, 'res_model_nm': str, 'rent_day': str,
                                    'rent_time': str, 'return_day': str, 'return_time': str, 'car_rent_fee': int,
                                    'cdw_fee': int, 'tot_fee': int, 'discount': float, 'res_day': str,
                                    'seasonality': int})

        # Change data types
        res_hx['rent_day'] = self._to_datetime(res_hx['rent_day'])
        res_hx['res_day'] = self._to_datetime(res_hx['res_day'])
        res_hx['discount'] = res_hx['discount'].astype(int)

        # self._prep_by_group(res_hx=res_hx)
        # Data preprocessing with car
        self._prep_by_car(res_hx=res_hx)

    def _prep_by_group(self, res_hx: pd.DataFrame):
        # Group car models (1.6 grade)
        res_hx_group = self._group_car_model(res_hx=res_hx)

        return res_hx_group

    def _prep_by_car(self, res_hx: pd.DataFrame):
        # Group by car
        res_hx = self._group_by_car(res_hx=res_hx)

        # Drop unncessary columns
        drop_cols = ['res_num', 'res_route_nm', 'res_model_nm', 'car_rent_fee', 'cdw_fee', 'tot_fee']
        res_hx = res_hx.drop(columns=drop_cols, errors='ignore')

        # Reorder dataframe
        res_hx = res_hx.sort_values(by=['rent_day', 'res_day'])

        # Make additional columns
        res_hx = self._add_der_variable(res_hx=res_hx)

        # Group by days
        res_hx_grp = self._grp_by(res_hx=res_hx)

    def _grp_by(self, res_hx: pd.DataFrame):
        self._grp_by_disc(res_hx=res_hx)
        self._grp_by_res_cnt(res_hx=res_hx)

    def _grp_by_disc(self, res_hx: pd.DataFrame):
        # Current discount
        disc_res_curr, disc_res_cum = self._grp_by_disc_res(res_hx=res_hx)

        # Cumulative discount
        res_hx_util = self._get_res_hx_util(res_hx=res_hx)
        self._grp_by_disc_util(res_hx=res_hx, res_hx_util=res_hx_util)

    def _get_res_hx_util(self, res_hx: pd.DataFrame):
        res_hx_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                res_hx['rent_day'], res_hx['rent_time'], res_hx['return_day'], res_hx['return_time'],
                res_hx['res_day'], res_hx['discount'], res_hx['res_model']):

            day_hour = timedelta(hours=24)
            six_hour = timedelta(hours=6)
            date_range = pd.date_range(start=rent_d, end=return_d)  # days of rent periods
            date_len = len(date_range)
            f = list(map(int, rent_t.split(':')))
            l = list(map(int, return_t.split(':')))
            ft = timedelta(hours=f[0], minutes=f[1])  # time of rent day
            lt = timedelta(hours=l[0], minutes=l[1])  # time of return day

            f_util = 1
            l_util = 1
            if (day_hour - ft) < six_hour:
                f_util = (day_hour - ft) / six_hour
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

            res_hx_util.extend(np.array([
                date_range,
                [res_day] * date_len,
                util,
                [discount] * date_len,
                [model] * date_len
            ]).T)

        res_hx_util_df = pd.DataFrame(res_hx_util,
                                      columns=['rent_day', 'res_day', 'util_rate', 'discount', 'res_model'])

        return res_hx_util_df

    def _grp_by_disc_res(self, res_hx: pd.DataFrame):
        # Reservation change on current discount
        disc_res_curr = self._grp_by_disc_res_curr(res_hx=res_hx)
        # Utilization change on current discount
        disc_res_cum = self._grp_by_disc_res_cum(res_hx=res_hx)

        return disc_res_curr, disc_res_cum

    def _grp_by_disc_util(self, res_hx: pd.DataFrame, res_hx_util: pd.DataFrame):
        disc_util_curr = self._grp_by_disc_util_curr(res_hx=res_hx, res_hx_util=res_hx_util)
        disc_util_cum = self._grp_by_disc_util_cum(res_hx=res_hx, res_hx_util=res_hx_util)

    @staticmethod
    def _grp_by_disc_res_curr(res_hx: pd.DataFrame):
        cnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt = cnt.rename('cnt_curr')
        dscnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()['discount']
        ss_lt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]
        disc_res_curr = pd.concat([ss_lt, cnt, dscnt], axis=1)
        disc_res_curr = disc_res_curr.reset_index(level=(0, 1, 2))

        return disc_res_curr

    @staticmethod
    def _grp_by_disc_res_cum(res_hx: pd.DataFrame):
        cnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        dscnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        dscnt_cum = dscnt.groupby(by=['rent_day', 'res_model']).cumsum()

        cum = pd.DataFrame({'cnt_cum': cnt_cum, 'dscnt_cum': dscnt_cum}, index=cnt_cum.index)
        cum['dscnt_mean'] = cum['dscnt_cum'] / cum['cnt_cum']

        ss_lt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]

        disc_res_cum = pd.concat([cum, ss_lt], axis=1)
        disc_res_cum = disc_res_cum.reset_index(level=(0, 1, 2))

        return disc_res_cum

    @staticmethod
    def _grp_by_disc_util_curr(res_hx: pd.DataFrame, res_hx_util: pd.DataFrame):
        util = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util_rate']
        util = util.rename('util_curr')

        dscnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        ss_lt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]
        disc_util_curr = pd.concat([ss_lt, util, dscnt], axis=1)
        disc_util_curr = disc_util_curr.reset_index(level=(0, 1, 2))

        return disc_util_curr

    def _grp_by_disc_util_cum(self, res_hx: pd.DataFrame, res_hx_util: pd.DataFrame):
        util = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util_rate']
        util = util.rename('util_curr')
        util_cum = util.groupby(by=['rent_day', 'res_model']).cumsum()

        dscnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        dscnt_cum = dscnt.groupby(by=['rent_day', 'res_model']).cumsum()

        ss_lt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]
        disc_util_cum = pd.concat([ss_lt, util_cum, dscnt_cum], axis=1)
        disc_util_cum = disc_util_cum.reset_index(level=(0, 1, 2))

        return disc_util_cum

    @staticmethod
    def _add_der_variable(res_hx: pd.DataFrame):
        # Add lead time
        res_hx['lead_time'] = res_hx['rent_day'] - res_hx['res_day']
        res_hx['lead_time'] = res_hx['lead_time'].to_numpy().astype('timedelta64[D]') / np.timedelta64(1, 'D')

        # Add lead time vector
        res_hx['lead_time_vec'] = (res_hx['lead_time'] // 7) + 24
        res_hx['lead_time_vec'] = np.where(res_hx['lead_time_vec'] < 28,
                                           res_hx['lead_time'].values,
                                           res_hx['lead_time_vec'].values)

        res_hx['lead_time'] = res_hx['lead_time'] * -1
        res_hx['lead_time_vec'] = res_hx['lead_time_vec'] * -1
        res_hx['lead_time'] = res_hx['lead_time'].astype(int)
        res_hx['lead_time_vec'] = res_hx['lead_time_vec'].astype(int)

        return res_hx

    @staticmethod
    def _group_by_car(res_hx: pd.DataFrame):
        av_ad = ['아반떼 AD (G)', '아반떼 AD (G) F/L']
        k3 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)']
        soul = ['쏘울 (G)', '쏘울 부스터 (G)']

        conditions = [
            res_hx['res_model_nm'].isin(av_ad),
            res_hx['res_model_nm'] == '올 뉴 아반떼 (G)',
            res_hx['res_model_nm'].isin(k3),
            res_hx['res_model_nm'].isin(soul),
            res_hx['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
        values = ['아반떼 AD (G) F/L', '올 뉴 아반떼 (G)', 'ALL NEW K3 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']

        res_hx['res_model'] = np.select(conditions, values)

        return res_hx

    def _make_res_hx(self):
        res_hx_17_19, res_hx_20, season_hx = self._load_data_hx()

        # Rename columns
        res_hx_17_19 = self._rename_col_hx(res_hx=res_hx_17_19)
        res_hx_20 = self._rename_col_hx(res_hx=res_hx_20)

        # Chnage data types
        res_hx_17_19['rent_day'] = self._to_datetime(arr=res_hx_17_19['rent_day'])
        # res_hx_17_19['rent_day'] = np.array(res_hx_17_19['rent_day'].to_numpy(), dtype='datetime64')
        res_hx_20['rent_day'] = self._to_datetime(arr=res_hx_20['rent_day'])
        # res_hx_20['rent_day'] = np.array(res_hx_20['rent_day'].to_numpy(), dtype='datetime64')
        season_hx['rent_day'] = self._to_datetime(arr=season_hx['rent_day'])
        # season_hx['rent_day'] = np.array(season_hx['rent_day'].to_numpy(), dtype='datetime64')

        # Filter timestamp
        res_hx_18_19 = res_hx_17_19[(res_hx_17_19['rent_day'] >= dt.datetime(2018, 1, 1)) &
                                    (res_hx_17_19['rent_day'] < dt.datetime(2020, 1, 1))]
        res_hx_20 = res_hx_20[res_hx_20['rent_day'] >= dt.datetime(2020, 1, 1)]

        # Merge dataset
        res_hx = self._merge_data_hx(concat_list=[res_hx_18_19, res_hx_20],
                                     merge_df=season_hx)

        # Filter dataset
        # Discount
        res_hx = res_hx[res_hx['discount'] != 100]
        # Car grade
        res_hx = res_hx[res_hx['res_model_nm'].isin(self.grade_1_6)]

        res_hx.to_csv(os.path.join(self.load_path, 'reservation', 'res_hx.csv'), index=False)

    def prep_curr(self):
        self.prep_res_curr()
        self.prep_res_util()

    def prep_res_curr(self):
        pass

    def prep_res_util(self):
        pass

    # History Reservation dataset
    def _load_data_hx(self):
        file_path_res_hx_17_19 = os.path.join(self.load_path, 'res_hx_17_19.csv')
        file_path_res_hx_20 = os.path.join(self.load_path, 'res_hx_20.csv')
        file_path_season_hx = os.path.join(self.load_path, 'seasonality', 'seasonality_hx.csv')
        data_type_res_hx = {'계약번호': int, '예약경로명': str, '예약모델명': str,
                            '대여일': str, '대여시간': str, '반납일': str, '반납시간': str,
                            '차량대여요금(VAT포함)': int, 'CDW요금': int, '총대여료(VAT포함)': int,
                            '적용할인율(%) ': int, '생성일': str}

        res_hx_17_19 = pd.read_csv(file_path_res_hx_17_19, delimiter='\t', dtype=data_type_res_hx)
        res_hx_20 = pd.read_csv(file_path_res_hx_20, delimiter='\t', dtype=data_type_res_hx)
        season_hx = pd.read_csv(file_path_season_hx)

        return res_hx_17_19, res_hx_20, season_hx

    @staticmethod
    def _rename_col_hx(res_hx: pd.DataFrame):
        rename_col_res_hx = {
            '계약번호': 'res_num', '예약경로명': 'res_route_nm', '예약모델명': 'res_model_nm',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '차량대여요금(VAT포함)': 'car_rent_fee', 'CDW요금': 'cdw_fee', '총대여료(VAT포함)': 'tot_fee',
            '적용할인율(%)': 'discount', '생성일': 'res_day'}

        return res_hx.rename(columns=rename_col_res_hx)

    @staticmethod
    def _to_datetime(arr: pd.Series):
        return pd.to_datetime(arr, format='%Y-%m-%d')

    @staticmethod
    def _merge_data_hx(concat_list: list, merge_df: pd.DataFrame):
        res_hx = pd.concat(concat_list)
        res_hx = res_hx.sort_values(by=['rent_day', 'res_day'])
        res_hx = res_hx.reset_index(drop=True)

        res_hx = pd.merge(res_hx, merge_df, how='left', on='rent_day', left_index=True, right_index=False)
        res_hx = res_hx.reset_index(drop=True)
        res_hx['seasonality'] = res_hx['seasonality'].fillna(0)

        return res_hx

    @staticmethod
    def _group_car_model(res_hx: pd.DataFrame):
        avante = ['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)']
        k3 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)']
        veloster_soul = ['더 올 뉴 벨로스터 (G)', '쏘울 (G)', '쏘울 부스터 (G)']

        conditions = [
            res_hx['res_model_nm'].isin(avante),
            res_hx['res_model_nm'].isin(k3),
            res_hx['res_model_nm'].isin(veloster_soul)]

        values = ['AVANTE', 'K3', 'VELOSTER']
        res_hx['res_model_grp'] = np.select(conditions, values)

    # Current Reservation dataset
    def _load_data_curr(self, update_date: str):
        file_path_res = os.path.join(self.load_path, 'reservation', 'res_' + update_date + '.csv')
        data_type = {'예약경로': int, '예약경로명': str, '계약번호': int, '고객구분': int,
                     '고객구분명': str, '총 청구액(VAT포함)': str, '예약모델': str, '예약모델명': str,
                     '차급': str, '대여일': str, '대여시간': str, '반납일': str, '반납시간': str,
                     '대여기간(일)': int, '대여기간(시간)': int, 'CDW요금': str, '할인유형': str,
                     '할인유형명': str, '적용할인명': str, '회원등급': str, '구매목적': str,
                     '생성일': str, '차종': str}
        res_curr = pd.rea_csv(file_path_res, delimiter='\t', dtype=data_type)

        file_path_capa = os.path.join(self.load_path, 'capa', 'capa_curr.csv')
        capa_curr = pd.read_csv(file_path_capa, delimiter='\t',
                                dtype={'date': str, 'model': str, 'capa': int})

