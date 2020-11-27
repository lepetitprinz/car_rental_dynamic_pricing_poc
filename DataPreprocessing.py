import os
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd


class DataPrep(object):

    def __init__(self):
        self.load_path = os.path.join('..', 'input')

        # History dataset
        self.season_hx: pd.DataFrame = pd.DataFrame()
        self.capa_hx_car: dict = dict()
        self.capa_hx_model: dict = dict()

        # Recent dataset
        self.season_re: pd.DataFrame = pd.DataFrame()
        self.capa_re_car: pd.DataFrame = pd.DataFrame()
        self.capa_re_model: pd.DataFrame = pd.DataFrame()

        self.capa: dict = dict()
        # car grade
        self.grade_1_6 = ['ALL NEW K3 (G)', '아반떼 AD (G)', '아반떼 AD (G) F/L',
                          '올 뉴 아반떼 (G)', '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        # self.grade_1_6 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)', '아반떼 AD (G)', '아반떼 AD (G) F/L',
        #                   '올 뉴 아반떼 (G)', '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']

    def prep_res_hx(self):
        # Make reservation history dataset
        # self._make_res_hx()

        # Load and set data
        res_hx = self._load_hx_dataset()

        # Filter models
        res_hx = res_hx[res_hx['res_model_nm'].isin(self.grade_1_6)]
        res_hx = res_hx.reset_index(drop=True)

        # Change data types
        res_hx['rent_day'] = self._to_datetime(res_hx['rent_day'])
        res_hx['res_day'] = self._to_datetime(res_hx['res_day'])
        res_hx['discount'] = res_hx['discount'].astype(int)

        # Data preprocessing: by car
        self._prep_by_group(df=res_hx, group='car', time='hx')

        # Data preprocessing: by model
        self._prep_by_group(df=res_hx, group='model', time='hx')

    def prep_res_recent(self, update_day: str):
        # Load ad set recent reservation dataset
        res_re = self._load_recent_dataset(update_day=update_day)

        # Rename columns
        res_re = self._rename_col_res_recent(df=res_re)

        # Filter grade
        res_re = res_re[res_re['res_model_nm'].isin(self.grade_1_6)]
        res_re = res_re.reset_index(drop=True)

        # Change data types
        res_re['rent_day'] = pd.to_datetime(res_re['rent_day'], format='%Y-%m-%d')
        res_re['res_day'] = pd.to_datetime(res_re['res_day'], format='%Y-%m-%d')

        # Merge dataset
        res_re = pd.merge(res_re, self.season_re, how='left', on='rent_day', left_index=True, right_index=False)
        res_re['seasonality'] = res_re['seasonality'].fillna(1)

        # Drop unnecessary columns
        res_re = self._drop_col_res_recent(df=res_re)

        # Data preprocessing: by car
        self._prep_by_group(df=res_re, group='car', time='re')
        # Data preprocessing: by model
        self._prep_by_group(df=res_re, group='model', time='re')

    def _drop_col_res_recent(self, df: pd.DataFrame):
        drop_cols = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm', 'tot_fee', 'res_model', 'car_grd',
                     'rent_period_day', 'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm',
                     'sale_purpose', 'applyed_discount', 'member_grd', 'sale_purpose', 'car_kind']

        return df.drop(columns=drop_cols, errors='ignore')

    def _rename_col_res_recent(self, df: pd.DataFrame):
        rename_cols = {
            '예약경로': 'res_route', '예약경로명': 'res_route_nm', '계약번호': 'res_num',
            '고객구분': 'cust_kind', '고객구분명': 'cust_kind_nm', '총 청구액(VAT포함)': 'tot_fee',
            '예약모델': 'res_model', '예약모델명': 'res_model_nm', '차급': 'car_grd',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '대여기간(일)': 'rent_period_day', '대여기간(시간)': 'rent_period_time',
            'CDW요금': 'cdw_fee', '할인유형': 'discount_type', '할인유형명': 'discount_type_nm',
            '적용할인명': 'applyed_discount', '적용할인율(%)': 'discount', '회원등급': 'member_grd',
            '구매목적': 'sale_purpose', '생성일': 'res_day', '차종': 'car_kind'}

        return df.rename(columns=rename_cols)

    def _load_recent_dataset(self, update_day: str):
        # Seasonal dataset
        data_path = os.path.join('..', 'input', 'seasonality', 'seasonality_curr.csv')
        season_recent = pd.read_csv(data_path, delimiter='\t', dtype={'date': str, 'seasonality': int})
        season_recent = season_recent.rename(columns={'date': 'rent_day'})
        season_recent['rent_day'] = pd.to_datetime(season_recent['rent_day'], format='%Y-%m-%d')
        self.season_re = season_recent

        # Capacity of models
        data_path = os.path.join('..', 'input', 'capa')
        capa_re_model = pd.read_csv(os.path.join(data_path, 'capa_curr_model.csv'), delimiter='\t',
                                    dtype={'date': str, 'model': str, 'capa': int})
        self.capa_re_model = {(month, model): capa for month, model, capa in zip(capa_re_model['date'],
                                                                                 capa_re_model['model'],
                                                                                 capa_re_model['capa'])}
        capa_re_car = pd.read_csv(os.path.join(data_path, 'capa_curr_car.csv'), delimiter='\t',
                                    dtype={'date': str, 'model': str, 'capa': int})
        self.capa_re_car = {(month, model): capa for month, model, capa in zip(capa_re_car['date'],
                                                                               capa_re_car['model'],
                                                                               capa_re_car['capa'])}

        self.capa = {'re': {'model': self.capa_re_model,
                            'car': self.capa_re_car}}

        # Recent reservation dataset
        data_path = os.path.join('..', 'input', 'reservation', 'res_' + update_day + '.csv')
        data_type = {'예약경로': int, '예약경로명': str, '계약번호': int, '고객구분': int, '고객구분명': str,
                     '총 청구액(VAT포함)': str, '예약모델': str, '예약모델명': str, '차급': str, '대여일': str,
                     '대여시간': str, '반납일': str, '반납시간': str, '대여기간(일)': int, '대여기간(시간)': int,
                     'CDW요금': str, '할인유형': str, '할인유형명': str, '적용할인명': str, '회원등급': str,
                     '구매목적': str, '생성일': str, '차종': str}

        return pd.read_csv(data_path, delimiter='\t', dtype=data_type)

    def _prep_by_group(self, df: pd.DataFrame, group: str, time: str):
        # Group by
        df = self._cluster_by_group(df=df, group=group)

        # Drop unncessary columns
        drop_cols = ['res_num', 'res_route_nm', 'res_model_nm', 'car_rent_fee', 'cdw_fee', 'tot_fee']
        df = df.drop(columns=drop_cols, errors='ignore')

        # Reorder dataframe
        df = df.sort_values(by=['rent_day', 'res_day'])

        # Make additional columns
        df = self._add_lead_time_vec(df=df)

        # Group
        self._grp_by_disc(df=df, group=group, time=time)

    def _load_hx_dataset(self):
        # Load reservation history
        res_hx = pd.read_csv(os.path.join(self.load_path, 'reservation', 'res_hx.csv'),
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
        self.capa = {'hx': {'model': self.capa_hx_model,
                            'car': self.capa_hx_car}}

        return res_hx

    def _grp_by_disc(self, df: pd.DataFrame, group: str, time: str):
        # Reservation
        disc_res_inc, disc_res_cum = self._grp_by_disc_cnt(res_cnt=df)

        # Utilization
        res_util = self._get_res_util(df=df)      # convert reservation to utilization dataset
        disc_util_inc, disc_util_cum = self._grp_by_disc_util(res_util=res_util, group=group, time=time)

        if time == 'hx':
            # Re-group
            re_grp = ['res_model', 'seasonality', 'lead_time_vec']
            disc_res_inc = disc_res_inc.groupby(by=re_grp).mean()
            disc_res_cum = disc_res_cum.groupby(by=re_grp).mean()
            disc_util_inc = disc_util_inc.groupby(by=re_grp).mean()
            disc_util_cum = disc_util_cum.groupby(by=re_grp).mean()

        elif time == 're':
            disc_res_inc = self._add_lead_time(disc_res_inc)
            disc_res_cum = self._add_lead_time(disc_res_cum)
            disc_util_inc = self._add_lead_time(disc_util_inc)
            disc_util_cum = self._add_lead_time(disc_util_cum)

        # Divide into each car model
        disc_res_inc_grp = self._div_into_group(df=disc_res_inc, group=group, time=time)
        disc_res_cum_grp = self._div_into_group(df=disc_res_cum, group=group, time=time)
        disc_util_inc_grp = self._div_into_group(df=disc_util_inc, group=group, time=time)
        disc_util_cum_grp = self._div_into_group(df=disc_util_cum, group=group, time=time)

        # Save each car model
        self._save_model(type_data=disc_res_inc_grp, type_name='cnt_inc', group=group, time=time)
        self._save_model(type_data=disc_res_cum_grp, type_name='cnt_cum', group=group, time=time)
        self._save_model(type_data=disc_util_inc_grp, type_name='util_inc', group=group, time=time)
        self._save_model(type_data=disc_util_cum_grp, type_name='util_cum', group=group, time=time)

    def _add_lead_time(self, df: pd.DataFrame):
        df['lead_time'] = df['rent_day'] - df['res_day']
        df['lead_time'] = df['lead_time'].to_numpy().astype('timedelta64[D]') / np.timedelta64(1, 'D')
        df['lead_time']= df['lead_time'] * -1

        return df

    def _set_capa(self, x, time, group):
        return self.capa[time][group][(x[0], x[1])]

    def _add_capacity(self, util: pd.DataFrame, group: str, time: str):
        util['month'] = util['rent_day'].dt.strftime('%Y%m')
        if group == 'car':
            util['capa'] = util[['month', 'res_model']].apply(self._set_capa, args=(time, group), axis=1)
        elif group == 'model':
            util['capa'] = util[['month', 'res_model']].apply(self._set_capa, args=(time, group), axis=1)
        return util

    @staticmethod
    def _div_into_group(df, group: str, time: str):
        groups = []
        if group == 'car':
            groups = ['아반떼 AD (G) F/L', '올 뉴 아반떼 (G)', 'ALL NEW K3 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        elif group == 'model':
            groups = ['AVANTE', 'K3', 'VELOSTER', 'SOUL']
        model_grp = {}
        for group in groups:
            if time == 'hx':
                model_grp[group] = df.loc[group].reset_index(level=(0, 1))
            elif time == 're':
                model_grp[group] = df[df['res_model'] == group]

        return model_grp

    @staticmethod
    def _save_model(type_data: dict, type_name: str, group: str, time: str):
        model_nm_map = {}
        if group == 'car':
            model_nm_map = {'아반떼 AD (G) F/L': 'av_ad', '올 뉴 아반떼 (G)': 'av_new', 'ALL NEW K3 (G)': 'k3',
                            '쏘울 부스터 (G)': 'soul', '더 올 뉴 벨로스터 (G)': 'vlst'}
        elif group == 'model':
            # model_nm_map = {'AVANTE': 'av', 'K3': 'k3', 'VELOSTER': 'vl'}
            model_nm_map = {'AVANTE': 'av', 'K3': 'k3', 'VELOSTER': 'vl', 'SOUL': 'su'}

        save_path = os.path.join('..', 'result', 'data', 'model_2', time, group)
        for model, data in type_data.items():
            data.to_csv(os.path.join(save_path, type_name, type_name + '_' + model_nm_map[model] + '.csv'), index=False)
            print(f'{model} of {type_name} data is saved.')

    @staticmethod
    def _get_res_util_BAK(df: pd.DataFrame):
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

    @staticmethod
    def _get_res_util(df: pd.DataFrame):
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
            lt = timedelta(hours=lst[0]+ 2, minutes=lst[1])  # time of return day

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
                date_range,
                [res_day] * date_len,
                util,
                [discount] * date_len,
                [model] * date_len
            ]).T)
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util_rate', 'discount', 'res_model'])

        return res_util_df

    def _grp_by_disc_cnt(self, res_cnt: pd.DataFrame):
        # Reservation change on current discount
        disc_res_inc = self._grp_by_disc_res_inc(res_cnt=res_cnt)
        # Utilization change on current discount
        disc_res_cum = self._grp_by_disc_res_cum(res_cnt=res_cnt)

        return disc_res_inc, disc_res_cum

    def _grp_by_disc_util(self, res_util: pd.DataFrame, group: str, time: str):
        # Increasing reservation count
        disc_util_inc = self._grp_by_disc_util_inc(res_util=res_util)
        # Cumulative reservation count
        disc_util_cum = self._grp_by_disc_util_cum(res_util=res_util)

        # Add seasonality
        disc_util_inc = self._merge_season(df=disc_util_inc, time=time)
        disc_util_cum = self._merge_season(df=disc_util_cum, time=time)

        # Add lead time vector
        disc_util_inc = self._add_lead_time_vec(df=disc_util_inc)
        disc_util_cum = self._add_lead_time_vec(df=disc_util_cum)

        # Add capacity and calculate utilization rate
        disc_util_inc = self._add_capacity(util=disc_util_inc, group=group, time=time)
        disc_util_cum = self._add_capacity(util=disc_util_cum, group=group, time=time)

        # Calculate utilization rate
        disc_util_inc['util_rate_add'] = disc_util_inc['util_add'] / disc_util_inc['capa']
        disc_util_cum['util_rate_cum'] = disc_util_cum['util_cum'] / disc_util_cum['capa']

        # Drop unnecessary columns
        disc_util_inc = disc_util_inc.drop(columns=['month', 'capa'])
        disc_util_cum = disc_util_cum.drop(columns=['month', 'capa'])

        return disc_util_inc, disc_util_cum

    def _merge_season(self, df: pd.DataFrame, time: str):
        if time == 'hx':
            return pd.merge(df, self.season_hx, how='left', on='rent_day', left_index=True, right_index=False)

        elif time == 're':
            return pd.merge(df, self.season_re, how='left', on='rent_day', left_index=True, right_index=False)

    @staticmethod
    def _grp_by_disc_res_inc(res_cnt: pd.DataFrame):
        # Group reservation counts
        cnt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt = cnt.rename('cnt_add')

        # Group discount rates
        disc = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).mean()['discount']
        ss_lt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]

        # Merge and reset index
        disc_res_inc = pd.concat([ss_lt, disc, cnt], axis=1)
        disc_res_inc = disc_res_inc.reset_index(level=(0, 1, 2))

        # Change data types
        disc_res_inc['seasonality'] = disc_res_inc['seasonality'].astype(int)
        disc_res_inc['lead_time_vec'] = disc_res_inc['lead_time_vec'].astype(int)

        return disc_res_inc

    @staticmethod
    def _grp_by_disc_res_cum(res_cnt: pd.DataFrame):
        # Group reservation counts
        cnt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        # Group discount rates
        disc = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        disc_cum = disc.groupby(by=['rent_day', 'res_model']).cumsum()

        cum = pd.DataFrame({'cnt_cum': cnt_cum, 'disc_cum': disc_cum}, index=cnt_cum.index)
        cum['disc_mean'] = cum['disc_cum'] / cum['cnt_cum']

        ss_lt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).mean()[['seasonality', 'lead_time_vec']]

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
    def _grp_by_disc_util_inc(res_util: pd.DataFrame):
        util = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util_rate']
        util = util.rename('util_add')

        disc = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).mean()['discount']
        disc_util_inc = pd.concat([disc, util], axis=1)
        disc_util_inc = disc_util_inc.reset_index(level=(0, 1, 2))

        return disc_util_inc

    @staticmethod
    def _grp_by_disc_util_cum(res_util: pd.DataFrame):
        cnt = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).count()['discount']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        util = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util_rate']
        util = util.rename('util_add')
        util_cum = util.groupby(by=['rent_day', 'res_model']).cumsum()

        disc = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        disc_cum = disc.groupby(by=['rent_day', 'res_model']).cumsum()

        disc_util_cum = pd.DataFrame({'disc_cum': disc_cum, 'util_cum': util_cum, 'cnt_cum': cnt_cum},
                                     index=util_cum.index)
        disc_util_cum['disc_mean'] = disc_util_cum['disc_cum'] / disc_util_cum['cnt_cum']

        disc_util_cum = disc_util_cum.reset_index(level=(0, 1, 2))

        # Drop unnecessary columns
        disc_util_cum = disc_util_cum.drop(columns=['cnt_cum', 'disc_cum'], errors='ignore')

        return disc_util_cum

    @staticmethod
    def _add_lead_time_vec(df: pd.DataFrame):
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
        # df['lead_time'] = df['lead_time'] * -1
        df['lead_time_vec'] = df['lead_time_vec'] * -1
        # df['lead_time'] = df['lead_time'].astype(int)
        df['lead_time_vec'] = df['lead_time_vec'].astype(int)
        df = df.drop(columns=['lead_time'])

        return df

    @staticmethod
    def _cluster_by_group(df: pd.DataFrame, group: str):
        if group == 'car':
            av_ad = ['아반떼 AD (G)', '아반떼 AD (G) F/L']
            # k3 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)']
            k3 = ['ALL NEW K3 (G)']
            soul = ['쏘울 (G)', '쏘울 부스터 (G)']

            conditions = [
                df['res_model_nm'].isin(av_ad),
                df['res_model_nm'] == '올 뉴 아반떼 (G)',
                df['res_model_nm'].isin(k3),
                df['res_model_nm'].isin(soul),
                df['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
            values = ['아반떼 AD (G) F/L', '올 뉴 아반떼 (G)', 'ALL NEW K3 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']

            df['res_model'] = np.select(conditions, values)

        elif group == 'model':
            av = ['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)']
            k3 = ['ALL NEW K3 (G)']
            # k3 = ['K3', 'THE NEW K3 (G)', 'ALL NEW K3 (G)']
            # vl = ['더 올 뉴 벨로스터 (G)', '쏘울 (G)', '쏘울 부스터 (G)']
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