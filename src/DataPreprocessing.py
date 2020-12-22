from Utility import Utility

import os
import datetime as dt
from dateutil.relativedelta import relativedelta

import pandas as pd


class DataPrep(object):

    def __init__(self, end_date: str, res_status_ud_day: str):
        self.utility = Utility
        self.load_path = self.utility.PATH_INPUT
        self.end_date = end_date
        self.res_status_ud_day = res_status_ud_day

        self.season: dict = {}
        self.capacity: dict = {}

        # car grade
        self.type_group: list = self.utility.TYPE_GROUP
        self.type_model: list = self.utility.TYPE_MODEL
        self.type_apply: dict = {'model': self.type_group, 'car': self.type_model}
        self.model_nm_map: dict = self.utility.MODEL_NAME_MAP

    def prep_res_history(self, type_apply: str):
        # Make reservation history dataset
        self._make_res_hx()

        # Load and set data
        res_hx = self.utility.load_res(time='hx')
        self.season['hx'] = self.utility.load_season(time='hx')
        self.capacity['hx'] = self._load_capa_hx(time='hx', type_apply=type_apply)

        # Filter models
        res_hx = self.utility.filter_model_grade(df=res_hx)

        # Change data types
        # string -> datetime
        res_hx['rent_day'] = pd.to_datetime(res_hx['rent_day'], format='%Y-%m-%d')
        res_hx['res_day'] = pd.to_datetime(res_hx['res_day'], format='%Y-%m-%d')
        # string -> integer
        res_hx['discount'] = res_hx['discount'].astype(int)

        # Data preprocessing
        self._prep_by_group(df=res_hx, type_apply=type_apply, time='hx')

        print("Data preprocessing for history data is finished")

    def prep_res_recent(self, res_status_ud_day: str, type_apply: str, time: str):
        # Load reservation status dataset
        res_re = self.utility.load_res(time=time, status_update_day=res_status_ud_day)
        # Load season and capacity dataset
        self.season['re'] = self.utility.load_season(time=time)
        self.capacity['re'] = self._load_capacity(time=time, type_apply=type_apply)

        # Rename columns
        res_re = res_re.rename(columns=self.utility.RENAME_COL_RES)
        # Filter grade of car models
        res_re = self.utility.filter_model_grade(df=res_re)

        # Change data types
        res_re['rent_day'] = pd.to_datetime(res_re['rent_day'], format='%Y-%m-%d')
        res_re['res_day'] = pd.to_datetime(res_re['res_day'], format='%Y-%m-%d')

        # Merge dataset
        res_re = pd.merge(res_re, self.season[time], how='left', on='rent_day', left_index=True, right_index=False)
        res_re['seasonality'] = res_re['seasonality'].fillna(1)

        # Drop unnecessary columns
        res_re = self._drop_col_res_recent(df=res_re)

        # Data preprocessing
        self._prep_by_group(df=res_re, type_apply=type_apply, time=time)
        print("Data preprocessing for recent data is finished")

        print('')
        print("Data preprocessing is finished.")
        print('')

    def _load_capa_hx(self, time: str, type_apply: str):
        # Load capacity of car models
        capacity = self.utility.load_capacity(time=time, type_apply=type_apply)
        # Convert monthly capacity to daily
        capacity = self._conv_mon_to_day_hx(df=capacity)
        # Mapping dictionary: (Data, Model) -> capacity
        capa_map = self.utility.make_capa_map(df=capacity)

        return capa_map

    def _load_capacity(self, time: str, type_apply: str):
        # Load capacity of car models
        capa_re = self.utility.load_capacity(time=time, type_apply=type_apply)
        # Load unavailable capacity of car models
        capa_re_unavail = self.utility.load_capacity(time=time, type_apply=type_apply, unavail=True)
        capa_re_unavail = capa_re_unavail.rename(columns={'capa': 'unavail'})
        # Convert monthly capacity to daily
        capa_re = self.utility.conv_mon_to_day(df=capa_re)
        # Subtract unavailable capacity
        capa_re = self.utility.apply_unavail_capa(capacity=capa_re, capa_unavail=capa_re_unavail)
        # Mapping dictionary: (Data, Model) -> capacity
        capa_map = self.utility.make_capa_map(df=capa_re)

        return capa_map

    ################################
    # Methods for grouping dataset
    ################################
    def _prep_by_group(self, df: pd.DataFrame, type_apply: str, time: str):
        # Cluster car models
        df = self.utility.cluster_model(df=df, type_apply=type_apply)

        # Drop unncessary columns
        drop_cols = ['res_num', 'res_route_nm', 'res_model_nm', 'car_rent_fee', 'cdw_fee', 'tot_fee']
        df = df.drop(columns=drop_cols, errors='ignore')

        # Reorder dataframe
        df = df.sort_values(by=['rent_day', 'res_day'])

        # Make additional columns
        df = self.utility.add_col_lead_time_vec(df=df)

        # Group
        grp_list = self._grp_by_disc(df=df, time=time)

        # Re-group
        re_grp_list = self._re_group(grp_list=grp_list, time=time)

        # Divide into each car model
        div_grp_list = self._div_model(grp_list=re_grp_list, type_apply=type_apply, time=time)

        # Save models
        self._save_each_model(results=div_grp_list, type_apply=type_apply, time=time)

    def _save_each_model(self, results: list, type_apply: str, time: str):
        type_names = ['cnt_inc', 'cnt_cum', 'util_inc', 'util_cum']
        for result, type_name in zip(results, type_names):
            self._save_model(type_data=result, type_name=type_name, type_apply=type_apply, time=time)

    def _div_model(self, grp_list: list, type_apply: str, time: str):
        div_list = []
        for df in grp_list:
            div_list.append(self._div_into_group(df=df, type_apply=type_apply, time=time))

        return div_list

    def _re_group(self, grp_list: list, time: str):
        re_grp_list = []
        for df in grp_list:
            if time == 'hx':
                re_grp = ['res_model', 'seasonality', 'lead_time_vec']
                re_grp_list.append(df.groupby(by=re_grp).mean())
            elif time == 're':
                re_grp_list.append(self.utility.add_col_lead_time(df))

        return re_grp_list

    def _grp_by_disc(self, df: pd.DataFrame, time: str):
        # Grouping reservation dataset
        disc_res_inc, disc_res_cum = self._grp_by_disc_cnt(res_cnt=df, time=time)

        # Calculate utilization
        res_util = self.utility.get_res_util(df=df)

        # Filter date (PoC Recommendation periods)
        end_date_dt = dt.datetime.strptime(''.join(self.end_date.split('/')), '%Y%m%d')
        res_util = res_util[res_util['rent_day'] <= end_date_dt]

        disc_util_inc, disc_util_cum = self._grp_by_disc_util(res_util=res_util, time=time)

        return [disc_res_inc, disc_res_cum, disc_util_inc, disc_util_cum]

    def _grp_by_disc_bak(self, df: pd.DataFrame, type_apply: str, time: str):
        # Grouping reservation dataset
        disc_res_inc, disc_res_cum = self._grp_by_disc_cnt(res_cnt=df, time=time)

        # Calculate utilization
        res_util = self.utility.get_res_util(df=df)

        # Filter date (PoC Recommendation periods)
        end_date_dt = dt.datetime.strptime(''.join(self.end_date.split('/')), '%Y%m%d')
        res_util = res_util[res_util['rent_day'] <= end_date_dt]

        disc_util_inc, disc_util_cum = self._grp_by_disc_util(res_util=res_util, time=time)

        if time == 'hx':
            # Re-group
            re_grp = ['res_model', 'seasonality', 'lead_time_vec']
            disc_res_inc = disc_res_inc.groupby(by=re_grp).mean()
            disc_res_cum = disc_res_cum.groupby(by=re_grp).mean()
            disc_util_inc = disc_util_inc.groupby(by=re_grp).mean()
            disc_util_cum = disc_util_cum.groupby(by=re_grp).mean()

        elif time == 're':
            disc_res_inc = self.utility.add_col_lead_time(disc_res_inc)
            disc_res_cum = self.utility.add_col_lead_time(disc_res_cum)
            disc_util_inc = self.utility.add_col_lead_time(disc_util_inc)
            disc_util_cum = self.utility.add_col_lead_time(disc_util_cum)

        # Divide into each car model
        disc_res_inc_grp = self._div_into_group(df=disc_res_inc, type_apply=type_apply, time=time)
        disc_res_cum_grp = self._div_into_group(df=disc_res_cum, type_apply=type_apply, time=time)
        disc_util_inc_grp = self._div_into_group(df=disc_util_inc, type_apply=type_apply, time=time)
        disc_util_cum_grp = self._div_into_group(df=disc_util_cum, type_apply=type_apply, time=time)

        # Save each car model
        self._save_model(type_data=disc_res_inc_grp, type_name='cnt_inc', type_apply=type_apply, time=time)
        self._save_model(type_data=disc_res_cum_grp, type_name='cnt_cum', type_apply=type_apply, time=time)
        self._save_model(type_data=disc_util_inc_grp, type_name='util_inc', type_apply=type_apply, time=time)
        self._save_model(type_data=disc_util_cum_grp, type_name='util_cum', type_apply=type_apply, time=time)

        print(f"Group: {type_apply} data preprocessing is finished")

    def _grp_by_disc_cnt(self, res_cnt: pd.DataFrame, time: str):
        # Add model capacity
        res_cnt = self._add_capacity(df=res_cnt, time=time)
        # Count divided by capacity
        res_cnt['cnt_rate'] = 1 / res_cnt['capa']

        # Reservation counts on current discount
        disc_res_inc = self._grp_by_disc_res_inc(res_cnt=res_cnt, time=time)
        # Cumulative reservation counts on current discount
        disc_res_cum = self._grp_by_disc_res_cum(res_cnt=res_cnt, time=time)

        return disc_res_inc, disc_res_cum

    @staticmethod
    def _grp_by_disc_res_inc(res_cnt: pd.DataFrame, time: str):
        # Group reservation counts
        cnt = None
        if time == 'hx':
            cnt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['cnt_rate']
        elif time == 're':
            cnt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']

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
    def _grp_by_disc_res_cum(res_cnt: pd.DataFrame, time: str):
        # Group on cumulative reservation counts
        cnt = None
        if time == 'hx':
            cnt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['cnt_rate']
        elif time == 're':
            cnt = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        # Group discount rate
        disc = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['discount']
        disc_cum = disc.groupby(by=['rent_day', 'res_model']).cumsum()

        cum = pd.DataFrame({'cnt_cum': cnt_cum, 'disc_cum': disc_cum}, index=cnt_cum.index)
        cnt_grp = res_cnt.groupby(by=['rent_day', 'res_model', 'res_day']).count()['lead_time_vec']
        cum['disc_mean'] = cum['disc_cum'] / cnt_grp.groupby(by=['rent_day', 'res_model']).cumsum()

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
    def _drop_col_res_recent(df: pd.DataFrame):
        drop_cols = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm', 'tot_fee', 'res_model', 'car_grd',
                     'rent_period_day', 'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm',
                     'sale_purpose', 'applied_discount', 'member_grd', 'sale_purpose', 'car_kind']

        return df.drop(columns=drop_cols, errors='ignore')

    @staticmethod
    def _conv_mon_to_day_hx(df: pd.DataFrame):
        df_days = pd.DataFrame()
        for yyyymm, model, capa in zip(df['date'], df['model'], df['capa']):
            dt_start = pd.to_datetime(yyyymm, format='%Y%m')
            dt_end = dt_start + relativedelta(months=+1) - dt.timedelta(days=1)
            days = pd.date_range(start=dt_start, end=dt_end)
            temp = pd.DataFrame({'date': days, 'model': model, 'capa': capa})
            df_days = pd.concat([df_days, temp], axis=0)

        return df_days

    def _set_capa(self, x, time):
        return self.capacity[time][(x[0], x[1])]

    def _add_capacity(self, df: pd.DataFrame, time: str):
        df['capa'] = df[['rent_day', 'res_model']].apply(self._set_capa, time=time, axis=1)

        return df

    def _div_into_group(self, df, type_apply: str, time: str):
        model_grp = {}
        for type_apply in self.type_apply[type_apply]:
            if time == 'hx':
                model_grp[type_apply] = df.loc[type_apply].reset_index(level=(0, 1))
            elif time == 're':
                model_grp[type_apply] = df[df['res_model'] == type_apply]

        return model_grp

    def _grp_by_disc_util(self, res_util: pd.DataFrame, time: str):
        # Increasing reservation count
        disc_util_inc = self._grp_by_disc_util_inc(res_util=res_util)
        # Cumulative reservation count
        disc_util_cum = self._grp_by_disc_util_cum(res_util=res_util)

        # Add seasonality
        disc_util_inc = self._merge_season(df=disc_util_inc, time=time)
        disc_util_cum = self._merge_season(df=disc_util_cum, time=time)

        # Add lead time vector
        disc_util_inc = self.utility.add_col_lead_time_vec(df=disc_util_inc)
        disc_util_cum = self.utility.add_col_lead_time_vec(df=disc_util_cum)

        # Add capacity and calculate utilization rate
        disc_util_inc = self._add_capacity(df=disc_util_inc, time=time)
        disc_util_cum = self._add_capacity(df=disc_util_cum, time=time)

        # Calculate utilization rate
        disc_util_inc['util_rate_add'] = disc_util_inc['util_add'] / disc_util_inc['capa']
        disc_util_cum['util_rate_cum'] = disc_util_cum['util_cum'] / disc_util_cum['capa']

        # Drop unnecessary columns
        disc_util_inc = disc_util_inc.drop(columns=['capa'], errors='ignore')
        disc_util_cum = disc_util_cum.drop(columns=['capa'], errors='ignore')

        return disc_util_inc, disc_util_cum

    def _merge_season(self, df: pd.DataFrame, time: str):
        return pd.merge(df, self.season[time], how='left', on='rent_day', left_index=True, right_index=False)

    @staticmethod
    def _grp_by_disc_util_inc(res_util: pd.DataFrame):
        util = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util']
        util = util.rename('util_add')

        disc = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).mean()['discount']
        disc_util_inc = pd.concat([disc, util], axis=1)
        disc_util_inc = disc_util_inc.reset_index(level=(0, 1, 2))

        return disc_util_inc

    @staticmethod
    def _grp_by_disc_util_cum(res_util: pd.DataFrame):
        cnt = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).count()['discount']
        cnt_cum = cnt.groupby(by=['rent_day', 'res_model']).cumsum()

        util = res_util.groupby(by=['rent_day', 'res_model', 'res_day']).sum()['util']
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

    def _make_res_hx(self):
        res_hx_17_19, res_hx_20 = self._load_data_hx()
        season_hx = self.utility.load_season(time='hx')

        # Rename columns
        res_hx_17_19 = self._rename_col_hx(res_hx=res_hx_17_19)
        res_hx_20 = self._rename_col_hx(res_hx=res_hx_20)

        # Chnage data types
        res_hx_17_19['rent_day'] = pd.to_datetime(res_hx_17_19['rent_day'], format='%Y-%m-%d')
        res_hx_20['rent_day'] = pd.to_datetime(res_hx_20['rent_day'], format='%Y-%m-%d')

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
        # Filter 1.6 grade of car models
        res_hx = self.utility.filter_model_grade(df=res_hx)

        res_hx.to_csv(os.path.join(self.load_path, 'res_status', 'res_hx.csv'), index=False)

    # History Reservation dataset
    def _load_data_hx(self):
        file_path_res_hx_17_19 = os.path.join(self.load_path, 'res_hx_17_19.csv')
        file_path_res_hx_20 = os.path.join(self.load_path, 'res_hx_20.csv')
        data_type_res_hx = {'계약번호': int, '예약경로명': str, '예약모델명': str,
                            '대여일': str, '대여시간': str, '반납일': str, '반납시간': str,
                            '차량대여요금(VAT포함)': int, 'CDW요금': int, '총대여료(VAT포함)': int,
                            '적용할인율(%) ': int, '생성일': str}

        res_hx_17_19 = pd.read_csv(file_path_res_hx_17_19, delimiter='\t', dtype=data_type_res_hx)
        res_hx_20 = pd.read_csv(file_path_res_hx_20, delimiter='\t', dtype=data_type_res_hx)

        return res_hx_17_19, res_hx_20

    @staticmethod
    def _rename_col_hx(res_hx: pd.DataFrame):
        rename_col_res_hx = {
            '계약번호': 'res_num', '예약경로명': 'res_route_nm', '예약모델명': 'res_model_nm',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '차량대여요금(VAT포함)': 'car_rent_fee', 'CDW요금': 'cdw_fee', '총대여료(VAT포함)': 'tot_fee',
            '적용할인율(%)': 'discount', '생성일': 'res_day'}

        return res_hx.rename(columns=rename_col_res_hx)

    @staticmethod
    def _merge_data_hx(concat_list: list, merge_df: pd.DataFrame):
        res_hx = pd.concat(concat_list)
        res_hx = res_hx.sort_values(by=['rent_day', 'res_day'])
        res_hx = res_hx.reset_index(drop=True)

        res_hx = pd.merge(res_hx, merge_df, how='left', on='rent_day', left_index=True, right_index=False)
        res_hx = res_hx.reset_index(drop=True)
        res_hx['seasonality'] = res_hx['seasonality'].fillna(1)

        return res_hx

    def _save_model(self, type_data: dict, type_name: str, type_apply: str, time: str):
        if time == 'hx':
            save_path = os.path.join('..', 'result', 'data', 'model_2', time)
        else:
            save_path = os.path.join('..', 'result', 'data', 'model_2', self.res_status_ud_day)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(os.path.join(save_path, type_apply)):
            os.mkdir(os.path.join(save_path, type_apply))
        if not os.path.exists(os.path.join(save_path, type_apply, type_name)):
            os.mkdir(os.path.join(save_path, type_apply, type_name))

        for model, data in type_data.items():
            data.to_csv(os.path.join(save_path, type_apply, type_name, ''.join([type_name, '_', model, '.csv'])),
                        index=False)
            print(f'{model} of {type_name} data is saved.')