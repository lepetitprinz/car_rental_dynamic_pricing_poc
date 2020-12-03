import os

import numpy as np
import pandas as pd

class DataPostProcessing(object):

    def __init__(self, res_update_day: str, cancel_update_day: str):
        self.res_update_day = res_update_day
        self.cancel_update_day = cancel_update_day
        self.res_re: pd.DataFrame = pd.DataFrame()
        self.cancel_re: pd.DataFrame = pd.DataFrame()
        # car grade
        self.grade_1_6 = ['ALL NEW K3 (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)',
                          '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        self.model_type = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
        self.status_cancel = ['취소', 'no show']
        self.drop_col_res = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm',
                            'res_model', 'car_grd', 'rent_time', 'return_day', 'return_time', 'rent_period_day',
                            'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm', 'sale_purpose',
                            'applied_discount', 'discount_rate', 'member_grd', 'sale_purpose', 'car_kind',
                            'res_model_nm']
        self.drop_col_cancel = ['return_datetime', 'res_channel', 'customer_number','tot_fee']


    def data_post_process(self):
        # Load dataset
        res_re, cancel_re = self._load_data()

        # Filter 1.6 grade cars
        res_re = res_re[res_re['res_model_nm'].isin(self.grade_1_6)]
        cancel_re = cancel_re[cancel_re['res_model_nm'].isin(self.grade_1_6)]

        # Filter canceled data
        cancel_re = cancel_re[cancel_re['status'].isin(self.status_cancel)]

        # Group
        res_re = self._group_model(df=res_re)
        cancel_re = self._group_model(df=cancel_re)

        # Drop Unnecessary columns
        res_re = res_re.drop(columns=self.drop_col_res, errors='ignore')
        cancel_re = cancel_re.drop(columns=self.drop_col_cancel, errors='ignore')

        # Chnage data types
        cancel_re['rent_datetime'] = pd.to_datetime(cancel_re['rent_datetime'], format='%Y-%m-%d %H:%M:%S')
        cancel_re['rent_day'] = cancel_re['rent_datetime'].dt.strftime('%Y%m%d')

        cancel_re_grp = cancel_re.groupby(by=['rent_day', 'res_model_grp']).count()['res_num']
        cancel_re_grp = cancel_re_grp.reset_index(level=(0, 1))

        print("")

    def _load_data(self):
        # Load recent reservation dataset
        data_path = os.path.join('..', 'input', 'reservation')
        res_re = pd.read_csv(os.path.join(data_path, 'res_' + self.res_update_day + '.csv'), delimiter='\t')

        # Rename columns
        res_remap_cols = {
            '예약경로': 'res_route', '예약경로명': 'res_route_nm', '계약번호': 'res_num',
            '고객구분': 'cust_kind', '고객구분명': 'cust_kind_nm', '총 청구액(VAT포함)': 'tot_fee',
            '예약모델': 'res_model', '예약모델명': 'res_model_nm', '차급': 'car_grd',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '대여기간(일)': 'rent_period_day', '대여기간(시간)': 'rent_period_time',
            'CDW요금': 'cdw_fee', '할인유형': 'discount_type', '할인유형명': 'discount_type_nm',
            '적용할인명': 'applied_discount', '적용할인율(%)': 'discount', '회원등급': 'member_grd',
            '구매목적': 'sale_purpose', '생성일': 'res_day', '차종': 'car_kind'}
        res_re = res_re.rename(columns=res_remap_cols)

        # Load recent reservation cancel dataset
        data_path = os.path.join('..', 'input', 'cancel')
        cancel_re = pd.read_csv(os.path.join(data_path, 'res_cancel_' + self.cancel_update_day + '.csv'), delimiter='\t')

        # Rename columns
        cancel_remap_cols = {
            '상태': 'status', '계약번호': 'res_num', '대여차종': 'res_model_nm',
            '대여일시': 'rent_datetime', '반납일시': 'return_datetime', '예약채널': 'res_channel',
            '고객번호': 'customer_number', '총청구액': 'tot_fee'}
        cancel_re = cancel_re.rename(columns=cancel_remap_cols)

        return res_re, cancel_re

    def _group_model(self, df: pd.DataFrame):
        # Group Car Model
        conditions = [
            df['res_model_nm'] == '아반떼 AD (G) F/L',
            df['res_model_nm'] == '올 뉴 아반떼 (G)',
            df['res_model_nm'] == 'ALL NEW K3 (G)',
            df['res_model_nm'].isin(['쏘울 (G)', '쏘울 부스터 (G)']),
            df['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
        values = self.model_type
        df['res_model_grp'] = np.select(conditions, values)

        return df


