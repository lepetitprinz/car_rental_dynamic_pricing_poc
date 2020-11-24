import os
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd


class DataPrep(object):

    def __init__(self):
        self.load_path = os.path.join('..', 'input')
        self.res_hx: pd.DataFrame = pd.DataFrame()
        self.season_hx: pd.DataFrame = pd.DataFrame()
        self.capa_hx_model: dict = dict()
        self.capa_hx_car: dict = dict()
        self.res_curr: pd.DataFrame = pd.DataFrame()
        self.capa_curr: pd.DataFrame = pd.DataFrame()

        # car grade
        self.grade_1_6 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)', '아반떼 AD (G)', '아반떼 AD (G) F/L',
                          '올 뉴 아반떼 (G)', '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']

    def prep_res_hx(self):
        # Make reservation history dataset
        # self._make_res_hx()

        self._set_data()

        # Change data types
        self.res_hx['rent_day'] = self._to_datetime(self.res_hx['rent_day'])
        self.res_hx['res_day'] = self._to_datetime(self.res_hx['res_day'])
        self.res_hx['discount'] = self.res_hx['discount'].astype(int)

        # Data preprocessing: by car
        # self._prep_by_group(res_hx=self.res_hx, group='car')
        # Data preprocessing: by model
        self._prep_by_group(res_hx=self.res_hx, group='model')

    def _prep_by_group(self, res_hx: pd.DataFrame, group: str):
        # Group by
        res_hx = self._cluster_by_group(res_hx=res_hx, group=group)

        # Drop unncessary columns
        drop_cols = ['res_num', 'res_route_nm', 'res_model_nm', 'car_rent_fee', 'cdw_fee', 'tot_fee']
        res_hx = res_hx.drop(columns=drop_cols, errors='ignore')

        # Reorder dataframe
        res_hx = res_hx.sort_values(by=['rent_day', 'res_day'])

        # Make additional columns
        res_hx = self._add_lead_time(res_hx=res_hx)

        # Group
        self._grp_by_disc(res_hx=res_hx, group=group)

    def _set_data(self):
        # Load reservation history
        self.res_hx = pd.read_csv(os.path.join(self.load_path, 'reservation', 'res_hx.csv'),
                                  dtype={'res_num': str, 'res_route_nm': str, 'res_model_nm': str, 'rent_day': str,
                                         'rent_time': str, 'return_day': str, 'return_time': str, 'car_rent_fee': int,
                                         'cdw_fee': int, 'tot_fee': int, 'discount': float, 'res_day': str,
                                         'seasonality': int})

        # Load seasonality history
        season_hx = pd.read_csv(os.path.join(self.load_path, 'seasonality', 'seasonality_hx.csv'), delimiter='\t')
        season_hx['rent_day'] = pd.to_datetime(season_hx['rent_day'], format='%Y-%m-%d')
        self.season_hx = season_hx

        # Load capacity history of car models
        capa_hx_path = os.path.join('..', 'input', 'capa')

        # Capacity of models
        capa_hx_model = pd.read_csv(os.path.join(capa_hx_path, 'capa_hx_model.csv'), delimiter='\t',
                                    dtype={'date': str, 'model': str, 'capa': int})
        self.capa_hx_model = {(month, model): capa for month, model, capa in zip(capa_hx_model['date'],
                                                                                 capa_hx_model['model'],
                                                                                 capa_hx_model['capa'])}

        # Capacity of cars
        capa_hx_car = pd.read_csv(os.path.join(capa_hx_path, 'capa_hx_car.csv'), delimiter='\t',
                                  dtype={'date': str, 'model': str, 'capa': int})
        self.capa_hx_car = {(month, model): capa for month, model, capa in zip(capa_hx_car['date'],
                                                                               capa_hx_car['model'],
                                                                               capa_hx_car['capa'])}


    def _grp_by_disc(self, res_hx: pd.DataFrame, group: str):
        # Reservation
        disc_res_inc, disc_res_cum = self._grp_by_disc_res(res_hx=res_hx)

        # Utilization
        res_hx_util = self._get_res_hx_util(res_hx=res_hx)      # convert reservation to utilization dataset
        disc_util_inc, disc_util_cum = self._grp_by_disc_util(res_hx_util=res_hx_util, group=group)

        # Re-group
        re_grp = ['res_model', 'seasonality', 'lead_time_vec']
        disc_res_inc_grp = disc_res_inc.groupby(by=re_grp).mean()
        disc_res_cum_grp = disc_res_cum.groupby(by=re_grp).mean()
        disc_util_inc_grp = disc_util_inc.groupby(by=re_grp).mean()
        disc_util_cum_grp = disc_util_cum.groupby(by=re_grp).mean()

        # Divide into each car model
        disc_res_inc_grp = self._div_into_group(df=disc_res_inc_grp, group=group)
        disc_res_cum_grp = self._div_into_group(df=disc_res_cum_grp, group=group)
        disc_util_inc_grp = self._div_into_group(df=disc_util_inc_grp, group=group)
        disc_util_cum_grp = self._div_into_group(df=disc_util_cum_grp, group=group)

        # Save each car model
        self._save_model(type_data=disc_res_inc_grp, type_name='disc_res_inc', group=group)
        self._save_model(type_data=disc_res_cum_grp, type_name='disc_res_cum', group=group)
        self._save_model(type_data=disc_util_inc_grp, type_name='disc_util_inc', group=group)
        self._save_model(type_data=disc_util_cum_grp, type_name='disc_util_cum', group=group)

    def set_capa_car(self, x):
        return self.capa_hx_car[(x[0], x[1])]

    def set_capa_model(self, x):
        return self.capa_hx_model[(x[0], x[1])]

    def _add_capacity(self, util: pd.DataFrame, group: str):
        util['month'] = util['rent_day'].dt.strftime('%Y%m')
        if group == 'car':
            util['capa'] = util[['month', 'res_model']].apply(self.set_capa_car, axis=1)
        elif group == 'model':
            util['capa'] = util[['month', 'res_model']].apply(self.set_capa_model, axis=1)
        return util

    @staticmethod
    def _div_into_group(df, group: str):
        if group == 'car':
            groups = ['아반떼 AD (G) F/L', '올 뉴 아반떼 (G)', 'ALL NEW K3 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        elif group == 'model':
            groups = ['AVANTE', 'K3', 'VELOSTER']
        model_grp = {}
        for group in groups:
            model_grp[group] = df.loc[group].reset_index(level=(0, 1))

        return model_grp

    @staticmethod
    def _save_model(type_data: dict, type_name: str, group: str):
        model_nm_map = {}
        if group == 'car':
            model_nm_map = {'아반떼 AD (G) F/L': 'av_ad', '올 뉴 아반떼 (G)': 'av_new', 'ALL NEW K3 (G)': 'k3',
                            '쏘울 부스터 (G)': 'soul', '더 올 뉴 벨로스터 (G)': 'vlst'}
        elif group == 'model':
            model_nm_map = {'AVANTE': 'av', 'K3': 'k3', 'VELOSTER': 'vl'}

        save_path = os.path.join('..', 'result', 'data', 'model_2', group)
        for model, data in type_data.items():
            data.to_csv(os.path.join(save_path, type_name + '_' + model_nm_map[model] + '.csv'), index=False)
            print(f'{model} of {type_name} data is saved.')

    @staticmethod
    def _get_res_hx_util(res_hx: pd.DataFrame):
        res_hx_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                res_hx['rent_day'], res_hx['rent_time'], res_hx['return_day'], res_hx['return_time'],
                res_hx['res_day'], res_hx['discount'], res_hx['res_model']):

            day_hour = timedelta(hours=24)
            six_hour = timedelta(hours=6)
            date_range = pd.date_range(start=rent_d, end=return_d)  # days of rent periods
            date_len = len(date_range)
            fst = list(map(int, rent_t.split(':')))
            lst = list(map(int, return_t.split(':')))
            ft = timedelta(hours=fst[0], minutes=fst[1])  # time of rent day
            lt = timedelta(hours=lst[0], minutes=lst[1])  # time of return day

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
        disc_res_inc = self._grp_by_disc_res_inc(res_hx=res_hx)
        # Utilization change on current discount
        disc_res_cum = self._grp_by_disc_res_cum(res_hx=res_hx)

        return disc_res_inc, disc_res_cum

    def _grp_by_disc_util(self, res_hx_util: pd.DataFrame, group: str):
        # Increasing reservation count
        disc_util_inc = self._grp_by_disc_util_inc(res_hx_util=res_hx_util)
        # Cumulative reservation count
        disc_util_cum = self._grp_by_disc_util_cum(res_hx_util=res_hx_util)

        # Add seasonality
        disc_util_inc = pd.merge(disc_util_inc, self.season_hx, how='left', on='rent_day',
                                 left_index=True, right_index=False)
        disc_util_cum = pd.merge(disc_util_cum, self.season_hx, how='left', on='rent_day',
                                 left_index=True, right_index=False)

        # Add lead time vector
        disc_util_inc = self._add_lead_time(res_hx=disc_util_inc)
        disc_util_cum = self._add_lead_time(res_hx=disc_util_cum)

        # Add capacity and calculate utilization rate
        disc_util_inc = self._add_capacity(util=disc_util_inc, group=group)
        disc_util_cum = self._add_capacity(util=disc_util_cum, group=group)

        # Calculate utilization rate
        disc_util_inc['util_rate_add'] = disc_util_inc['util_add'] / disc_util_inc['capa']
        disc_util_cum['util_rate_cum'] = disc_util_cum['util_cum'] / disc_util_cum['capa']

        # Drop unnecessary columns
        disc_util_inc = disc_util_inc.drop(columns=['month', 'capa', 'util_add'])
        disc_util_cum = disc_util_cum.drop(columns=['month', 'capa', 'util_cum'])

        return disc_util_inc, disc_util_cum

    @staticmethod
    def _grp_by_disc_res_inc(res_hx: pd.DataFrame):
        # Group reservation counts
        cnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt = cnt.rename('cnt_add')

        # Group discount rates
        disc = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()['discount']
        ss_lt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]

        # Merge and reset index
        disc_res_inc = pd.concat([ss_lt, disc, cnt], axis=1)
        disc_res_inc = disc_res_inc.reset_index(level=(0, 1, 2))

        # Change data types
        disc_res_inc['seasonality'] = disc_res_inc['seasonality'].astype(int)
        disc_res_inc['lead_time_vec'] = disc_res_inc['lead_time_vec'].astype(int)

        return disc_res_inc

    @staticmethod
    def _grp_by_disc_res_cum(res_hx: pd.DataFrame):
        # Group reservation counts
        cnt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        # Group discount rates
        disc = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        disc_cum = disc.groupby(by=['rent_day', 'res_model']).cumsum()

        cum = pd.DataFrame({'cnt_cum': cnt_cum, 'disc_cum': disc_cum}, index=cnt_cum.index)
        cum['disc_mean'] = cum['disc_cum'] / cum['cnt_cum']

        ss_lt = res_hx.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]

        # Merge and reset index
        disc_res_cum = pd.concat([ss_lt, cum], axis=1)
        disc_res_cum = disc_res_cum.reset_index(level=(0, 1, 2))

        # Drop unnecessary columns
        disc_res_cum = disc_res_cum.drop(columns=['disc_cum'])

        # Change data types
        disc_res_cum['seasonality'] = disc_res_cum['seasonality'].astype(int)
        disc_res_cum['lead_time_vec'] = disc_res_cum['lead_time_vec'].astype(int)

        return disc_res_cum

    @staticmethod
    def _grp_by_disc_util_inc(res_hx_util: pd.DataFrame):
        util = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util_rate']
        util = util.rename('util_add')

        disc = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).mean()['discount']
        disc_util_inc = pd.concat([disc, util], axis=1)
        disc_util_inc = disc_util_inc.reset_index(level=(0, 1, 2))

        return disc_util_inc

    @staticmethod
    def _grp_by_disc_util_cum(res_hx_util: pd.DataFrame):
        cnt = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).count()['discount']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        util = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util_rate']
        util = util.rename('util_add')
        util_cum = util.groupby(by=['rent_day', 'res_model']).cumsum()

        disc = res_hx_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        disc_cum = disc.groupby(by=['rent_day', 'res_model']).cumsum()

        disc_util_cum = pd.DataFrame({'disc_cum': disc_cum, 'util_cum': util_cum, 'cnt_cum': cnt_cum},
                                     index=util_cum.index)
        disc_util_cum['disc_mean'] = disc_util_cum['disc_cum'] / disc_util_cum['cnt_cum']

        disc_util_cum = disc_util_cum.reset_index(level=(0, 1, 2))

        # Drop unnecessary columns
        disc_util_cum = disc_util_cum.drop(columns=['cnt_cum', 'disc_cum'], errors='ignore')

        return disc_util_cum

    @staticmethod
    def _add_lead_time(res_hx: pd.DataFrame):
        # Add lead time
        res_hx['lead_time'] = res_hx['rent_day'] - res_hx['res_day']
        res_hx['lead_time'] = res_hx['lead_time'].to_numpy().astype('timedelta64[D]') / np.timedelta64(1, 'D')

        # Add lead time vector
        res_hx['lead_time_vec'] = (res_hx['lead_time'] // 7) + 24       # Lead time > 28 : 1 week
        res_hx['lead_time_vec'] = np.where(res_hx['lead_time_vec'] < 28,    # Lead time 1 ~ 27
                                           res_hx['lead_time'].values,
                                           res_hx['lead_time_vec'].values)

        # Change data type
        res_hx['lead_time'] = res_hx['lead_time'] * -1
        res_hx['lead_time_vec'] = res_hx['lead_time_vec'] * -1
        res_hx['lead_time'] = res_hx['lead_time'].astype(int)
        res_hx['lead_time_vec'] = res_hx['lead_time_vec'].astype(int)

        return res_hx

    @staticmethod
    def _cluster_by_group(res_hx: pd.DataFrame, group: str):
        if group == 'car':
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

        elif group == 'model':
            av = ['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)']
            k3 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)']
            vl = ['더 올 뉴 벨로스터 (G)', '쏘울 (G)', '쏘울 부스터 (G)']

            conditions = [
                res_hx['res_model_nm'].isin(av),
                res_hx['res_model_nm'].isin(k3),
                res_hx['res_model_nm'].isin(vl)]

            values = ['AVANTE', 'K3', 'VELOSTER']
            res_hx['res_model'] = np.select(conditions, values)

        return res_hx

    def _make_res_hx(self):
        res_hx_17_19, res_hx_20, season_hx = self._load_data_hx()

        # Rename columns
        res_hx_17_19 = self._rename_col_hx(res_hx=res_hx_17_19)
        res_hx_20 = self._rename_col_hx(res_hx=res_hx_20)

        # Chnage data types
        res_hx_17_19['rent_day'] = self._to_datetime(arr=res_hx_17_19['rent_day'])
        res_hx_20['rent_day'] = self._to_datetime(arr=res_hx_20['rent_day'])
        season_hx['rent_day'] = self._to_datetime(arr=season_hx['rent_day'])

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
        season_hx = pd.read_csv(file_path_season_hx, delimiter='\t')

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
        res_hx['res_model'] = np.select(conditions, values)

        return res_hx

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

