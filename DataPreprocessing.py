import os
import datetime as dt

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

        # Delete unnecessary variables
        del res_hx_17_19
        del res_hx_18_19
        del res_hx_20

        # Change data types
        res_hx['res_day'] = self._to_datetime(res_hx['res_day'])
        res_hx['discount'] = res_hx['discount'].astype(int)
        res_hx['seasonality'] = res_hx['seasonality'].astype(int)

        # Filter dataset
        # Discount
        res_hx = res_hx[res_hx['discount'] != 100]
        # Car grade
        res_hx = res_hx[res_hx['res_model_nm'].isin(self.grade_1_6)]

        self._prep_by_group(res_hx=res_hx)
        self._prep_by_car(res_hx=res_hx)

    def _prep_by_group(self, res_hx: pd.DataFrame):
        # Group car models (1.6 grade)
        res_hx_group = self._group_car_model(res_hx=res_hx)

    def _prep_by_car(self, res_hx: pd.DataFrame):
        pass

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
    def _del_unnec_variable(var_list: list):
        for var in var_list:
            del var

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

