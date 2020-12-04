import os
import datetime as dt
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd


class DataPostProcessing(object):

    def __init__(self, update_day_before: str, update_day_after: str, update_day_cancel: str,
                 res_complete_day: str, disc_rec_day: str, start_day: str, end_day: str):
        # Path
        self.path_hx_data = os.path.join('..', 'result', 'data', 'model_2', 'hx', 'car')
        # Date
        self.update_day_before = update_day_before
        self.update_day_after = update_day_after
        self.update_day_cancel = update_day_cancel
        self.res_complete_day = res_complete_day
        self.disc_rec_day = disc_rec_day
        self.start_day = start_day
        self.end_day = end_day
        # Data
        self.res_re: pd.DataFrame = pd.DataFrame()
        self.cancel_re: pd.DataFrame = pd.DataFrame()
        self.capa_re: dict = {}
        self.disc_re: dict = {}
        self.disc_rec: dict = {}
        # car grade
        self.grade_1_6 = ['ALL NEW K3 (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)',
                          '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        self.model_type = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
        self.model_type_map_rev = {'아반떼 AD (G) F/L': 'av_ad', '올 뉴 아반떼 (G)': 'av_new',
                                   'ALL NEW K3 (G)': 'k3', '쏘울 부스터 (G)': 'soul',
                                   '더 올 뉴 벨로스터 (G)': 'vlst'}
        self.model_grp = {'av_ad': 'av', 'av_new': 'av', 'k3': 'k3', 'soul': 'su', 'vlst': 'vl'}
        self.status_cancel = ['취소', 'no show']
        self.drop_col_res = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm',
                             'res_model', 'car_grd', 'rent_period_day', 'rent_period_time', 'cdw_fee',
                             'discount_type', 'discount_type_nm', 'sale_purpose', 'applied_discount', 'discount_rate',
                             'member_grd', 'sale_purpose', 'car_kind', 'res_model_nm']
        self.drop_col_cancel = ['return_datetime', 'res_channel', 'customer_number', 'tot_fee',
                                'res_model_nm', 'status']

    def post_process(self):
        # Load dataset
        res_before, res_after, res_cancel = self._load_data()

        #
        self.capa_re = self._get_capa_re()
        self.disc_re = self._get_disc_re()
        self.disc_rec = self._get_disc_rec()

        # Change data type
        res_before['rent_day'] = pd.to_datetime(res_before['rent_day'], format='%Y-%m-%d')
        res_after['rent_day'] = pd.to_datetime(res_after['rent_day'], format='%Y-%m-%d')
        res_cancel['rent_datetime'] = pd.to_datetime(res_cancel['rent_datetime'], format='%Y-%m-%d %H:%M:%S')
        res_cancel['rent_day'] = res_cancel['rent_datetime'].dt.strftime('%Y%m%d')
        res_cancel['rent_day'] = pd.to_datetime(res_cancel['rent_day'], format='%Y%m%d')

        # Filter datetime
        ud_day_bf_dt = dt.datetime.strptime(self.update_day_before, '%y%m%d')
        start_day_dt = dt.datetime.strptime(self.start_day, '%Y%m%d')
        end_day_dt = dt.datetime.strptime(self.end_day, '%Y%m%d')
        res_before = res_before[(res_before['rent_day'] >= ud_day_bf_dt) & (res_before['rent_day'] <= end_day_dt)]
        res_after = res_after[(res_after['rent_day'] >= ud_day_bf_dt) & (res_after['rent_day'] <= end_day_dt)]

        # Filter 1.6 grade cars
        res_before = res_before[res_before['res_model_nm'].isin(self.grade_1_6)]
        res_after = res_after[res_after['res_model_nm'].isin(self.grade_1_6)]
        res_cancel = res_cancel[res_cancel['res_model_nm'].isin(self.grade_1_6)]

        # Filter canceled data
        res_cancel = res_cancel[res_cancel['status'].isin(self.status_cancel)]

        # Group
        res_before = self._group_model(df=res_before)
        res_after = self._group_model(df=res_after)
        res_cancel = self._group_model(df=res_cancel)

        # Drop Unnecessary columns
        res_before = res_before.drop(columns=self.drop_col_res, errors='ignore')
        res_after = res_after.drop(columns=self.drop_col_res, errors='ignore')
        res_cancel = res_cancel.drop(columns=self.drop_col_cancel, errors='ignore')

        # Reservation count
        cnt_bf = res_before.groupby(by=['model', 'rent_day']).count()['res_num']
        cnt_af = res_after.groupby(by=['model', 'rent_day']).count()['res_num']
        cnt_cancel = res_cancel.groupby(by=['rent_day', 'model']).count()['res_num']
        cnt_bf = cnt_bf.reset_index(level=(0, 1))
        cnt_af = cnt_af.reset_index(level=(0, 1))
        cnt_cancel = cnt_cancel.reset_index(level=(0, 1))
        cnt_cancel = cnt_cancel.rename(columns={'res_num': 'canceled'})

        # Change to utilization rate
        util_bf = self._get_res_util(df=res_before)
        util_af = self._get_res_util(df=res_after)

        util_bf_grp = util_bf.groupby(by=['model', 'rent_day']).sum()['util']
        util_af_grp = util_af.groupby(by=['model', 'rent_day']).sum()['util']
        util_bf_grp = util_bf_grp.reset_index(level=(0, 1))
        util_af_grp = util_af_grp.reset_index(level=(0, 1))

        util_bf_grp['rent_mon'] = util_bf_grp['rent_day'].dt.strftime('%Y%m')
        util_af_grp['rent_mon'] = util_af_grp['rent_day'].dt.strftime('%Y%m')
        util_bf_grp['util_rate'] = util_bf_grp[['rent_mon', 'model', 'util']].apply(self._calc_util_rate, axis=1)
        util_af_grp['util_rate'] = util_af_grp[['rent_mon', 'model', 'util']].apply(self._calc_util_rate, axis=1)
        util_bf_grp = util_bf_grp.drop(columns=['rent_mon'])
        util_af_grp = util_af_grp.drop(columns=['rent_mon'])

        # scaling
        util_bf_grp['util'] = np.round(util_bf_grp['util'].to_numpy(), 1)
        util_af_grp['util'] = np.round(util_af_grp['util'].to_numpy(), 1)
        util_bf_grp['util_rate'] = np.round(util_bf_grp['util_rate'].to_numpy() * 100, 1)
        util_af_grp['util_rate'] = np.round(util_af_grp['util_rate'].to_numpy() * 100, 1)

        # Reservation applying discount
        disc_bf = util_bf.groupby(by=['model', 'rent_day']).mean()['discount']
        disc_af = util_af.groupby(by=['model', 'rent_day']).mean()['discount']
        disc_bf = disc_bf.reset_index(level=(0, 1))
        disc_af = disc_af.reset_index(level=(0, 1))

        # Scaling
        disc_bf['discount'] = np.round(disc_bf['discount'].to_numpy(), 1)
        disc_af['discount'] = np.round(disc_af['discount'].to_numpy(), 1)

        # Merge

        result_bf = pd.merge(util_bf_grp, disc_bf, how='left', on=['model', 'rent_day'],
                             left_index=True, right_index=False)
        result_bf = pd.merge(result_bf, cnt_bf, how='left', on=['model', 'rent_day'],
                             left_index=True, right_index=False)

        result_af = pd.merge(util_af_grp, disc_af, how='left', on=['model', 'rent_day'],
                             left_index=True, right_index=False)
        result_af = pd.merge(result_af, cnt_af, how='left', on=['model', 'rent_day'],
                             left_index=True, right_index=False)

        # Rename columns
        result_bf = result_bf.rename(columns={'res_num': 'cnt_bf', 'util': 'util_bf', 'util_rate': 'util_rate_bf',
                                              'discount': 'disc_bf'})
        result_af = result_af.rename(columns={'res_num': 'cnt_af', 'util': 'util_af', 'util_rate': 'util_rate_af',
                                              'discount': 'disc_af'})

        result = pd.merge(result_bf, result_af, how='outer', on=['model', 'rent_day'],
                          left_index=True, right_index=False)
        result = pd.merge(result, cnt_cancel, how='left', on=['model', 'rent_day'],
                          left_index=True, right_index=False)

        # Filter days
        result = result[(result['rent_day'] >= start_day_dt) & (result['rent_day'] <= end_day_dt)]

        # Fill NA
        result = result.fillna(0)

        # Dict
        result_dict = {}
        date_range = pd.date_range(start=self.start_day, end=self.end_day, freq='D')
        rearr = ['rent_day',
                 'cnt_af', 'cnt_exp', 'cnt_chg',
                 'util_af', 'util_exp', 'util_chg',
                 'util_rate_af', 'util_rate_exp', 'util_rate_chg',
                 'disc_standard', 'disc_rec', 'disc_bf', 'disc_af', 'canceled']
        for model in self.model_type:
            date_df = pd.DataFrame({'rent_day': date_range})
            temp = result[result['model'] == model].sort_values(by='rent_day')
            temp = temp.reset_index(drop=True)
            temp = pd.merge(date_df, temp, how='left', on=['rent_day'], left_index=True, right_index=False)
            temp = temp.fillna(0)
            temp['disc_standard'] = [self.disc_re[rent_day] for rent_day in temp['rent_day']]
            temp['cnt_exp'] = [self.disc_rec[self.model_grp[model]][rent_day][0] for rent_day in temp['rent_day']]
            temp['cnt_chg'] = temp['cnt_af'] - temp['cnt_exp']
            temp['util_exp'] = [self.disc_rec[self.model_grp[model]][rent_day][1] for rent_day in temp['rent_day']]
            temp['util_chg'] = temp['util_af'] - temp['util_exp']
            temp['util_rate_exp'] = [self.disc_rec[self.model_grp[model]][rent_day][2] for rent_day in temp['rent_day']]
            temp['util_rate_chg'] = temp['util_rate_af'] - temp['util_rate_exp']
            temp['disc_rec'] = [self.disc_rec[self.model_grp[model]][rent_day][3] for rent_day in temp['rent_day']]
            temp = temp[rearr]
            result_dict[model] = temp

        save_path = os.path.join('..', 'result', 'data', 'weekly_report', self.update_day_after)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for model, df in result_dict.items():
            df['rent_day'] = df['rent_day'].dt.strftime('%Y-%m-%d')
            df.T.to_csv(os.path.join(save_path, 'weekly_report_' + model + '.csv'), header=False)

        print("")

    def _calc_util_rate(self, x):
        return x[2] / self.capa_re[(x[0], x[1])]

    def _load_data(self):
        # Load recent reservation dataset
        data_path = os.path.join('..', 'input', 'reservation')
        res_before = pd.read_csv(os.path.join(data_path, 'res_' + self.update_day_before + '.csv'), delimiter='\t')
        res_after = pd.read_csv(os.path.join(data_path, 'res_' + self.update_day_after + '.csv'), delimiter='\t')
        data_path = os.path.join('..', 'input', 'res_complete')
        res_complete = pd.read_csv(os.path.join(data_path, 'res_complete_' + self.res_complete_day + '.csv'),
                                   delimiter='\t')

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
        res_before = res_before.rename(columns=res_remap_cols)
        res_after = res_after.rename(columns=res_remap_cols)
        res_complete = res_complete.rename(columns=res_remap_cols)

        # Data Preprocessing
        res_complete['res_route'] = 0    # Exception
        res_complete = res_complete[res_complete['sale_purpose'] == '단기']
        res_after = pd.concat([res_after, res_complete], axis=0)

        # Load recent reservation cancel dataset
        data_path = os.path.join('..', 'input', 'cancel')
        cancel_re = pd.read_csv(os.path.join(data_path, 'res_cancel_' + self.update_day_cancel + '.csv'), delimiter='\t')

        # Rename columns
        cancel_remap_cols = {
            '상태': 'status', '계약번호': 'res_num', '대여차종': 'res_model_nm',
            '대여일시': 'rent_datetime', '반납일시': 'return_datetime', '예약채널': 'res_channel',
            '고객번호': 'customer_number', '총청구액': 'tot_fee'}
        cancel_re = cancel_re.rename(columns=cancel_remap_cols)

        return res_before, res_after, cancel_re

    def _get_capa_re(self):
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'capa')
        capa_init = pd.read_csv(os.path.join(load_path, 'capa_curr_car.csv'), delimiter='\t',
                                dtype={'date': str, 'model': str, 'capa': int})
        capa_re = {(date, self.model_type_map_rev[model]): capa for date, model, capa in zip(capa_init['date'],
                                                                                             capa_init['model'],
                                                                                             capa_init['capa'])}

        return capa_re

    def _get_disc_re(self):
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'disc_complete')
        disc_re = pd.read_csv(os.path.join(load_path, 'disc_complete_' + self.update_day_after + '.csv'), delimiter='\t',
                                    dtype={'date': str, 'disc': int})
        disc_re['date'] = pd.to_datetime(disc_re['date'], format='%Y%m%d')
        disc_re = {date: disc for date, disc in zip(disc_re['date'], disc_re['disc'])}

        return disc_re

    def _get_disc_rec(self):
        disc_rec = {}
        load_path = os.path.join('..', 'result', 'data', 'recommend', 'summary', self.disc_rec_day, 'original')
        for model_grp in ['av', 'k3', 'su', 'vl']:
            df = pd.read_csv(os.path.join(load_path, 'rec_summary_' + model_grp + '.csv'), encoding='euc-kr')
            df = df[['대여일', '기대 예약건수', '기대 가동대수(시간)', '기대가동률', '추천 할인율']]
            df = df.rename(columns={'대여일': 'rent_day', '기대 예약건수': 'exp_cnt', '기대 가동대수(시간)': 'exp_util',
                                    '기대가동률': 'exp_util_rate', '추천 할인율': 'disc_rec'})
            df['rent_day'] = pd.to_datetime(df['rent_day'], format='%Y-%m-%d')
            disc_rec[model_grp] = {date: [cnt, util, util_rate, disc] for date, cnt,
                                   util, util_rate, disc in zip(df['rent_day'], df['exp_cnt'], df['exp_util'],
                                                                df['exp_util_rate'], df['disc_rec'])}

        return disc_rec

    def _group_model(self, df: pd.DataFrame):
        # Group Car Model
        conditions = [
            df['res_model_nm'] == '아반떼 AD (G) F/L',
            df['res_model_nm'] == '올 뉴 아반떼 (G)',
            df['res_model_nm'] == 'ALL NEW K3 (G)',
            df['res_model_nm'] == '쏘울 부스터 (G)',
            df['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
        values = self.model_type
        df['model'] = np.select(conditions, values)

        return df

    @staticmethod
    def _get_res_util(df: pd.DataFrame):
        res_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                df['rent_day'], df['rent_time'], df['return_day'], df['return_time'],
                df['res_day'], df['discount'], df['model']):

            day_hour = timedelta(hours=24)
            six_hour = timedelta(hours=6)
            date_range = pd.date_range(start=rent_d, end=return_d)  # days of rent periods
            date_len = len(date_range)
            fst = list(map(int, rent_t.split(':')))
            lst = list(map(int, return_t.split(':')))
            ft = timedelta(hours=fst[0], minutes=fst[1])      # time of rent day
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
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util', 'discount', 'model'])

        return res_util_df



    def _load_hx_data(self):
        # Load Reservation Count dataset
        self.cnt_av_ad = pd.read_csv(os.path.join(self.path_hx_data, 'cnt_cum', 'cnt_cum_av_ad.csv'))
        self.cnt_av_new = pd.read_csv(os.path.join(self.path_hx_data, 'cnt_cum', 'cnt_cum_av_new.csv'))
        self.cnt_k3 = pd.read_csv(os.path.join(self.path_hx_data, 'cnt_cum', 'cnt_cum_k3.csv'))
        self.cnt_vlst = pd.read_csv(os.path.join(self.path_hx_data, 'cnt_cum', 'cnt_cum_vlst.csv'))
        self.cnt_soul = pd.read_csv(os.path.join(self.path_hx_data, 'cnt_cum', 'cnt_cum_soul.csv'))

        # Load Reservation Utilization dataset
        self.res_util_av_ad = pd.read_csv(os.path.join(self.path_hx_data, 'util_cum', 'util_cum_av_ad.csv'))
        self.res_util_av_new = pd.read_csv(os.path.join(self.path_hx_data, 'util_cum', 'util_cum_av_new.csv'))
        self.res_util_k3 = pd.read_csv(os.path.join(self.path_hx_data, 'util_cum', 'util_cum_k3.csv'))
        self.res_util_vlst = pd.read_csv(os.path.join(self.path_hx_data, 'util_cum', 'util_cum_vlst.csv'))
        self.res_util_soul = pd.read_csv(os.path.join(self.path_hx_data, 'util_cum', 'util_cum_soul.csv'))

    def _set_split_map(self):
        data_map = {'cnt': {'av_ad': self.cnt_av_ad,
                            'av_new': self.cnt_av_new,
                            'k3': self.cnt_k3,
                            'vl': self.cnt_vlst,
                            'su': self.cnt_soul},
                    'disc': {'av_ad': self.cnt_av_ad,
                             'av_new': self.cnt_av_new,
                             'k3': self.cnt_k3,
                             'vl': self.cnt_vlst,
                             'su': self.cnt_soul},
                    'util': {'av_ad': self.res_util_av_ad,
                             'av_new': self.res_util_av_new,
                             'k3': self.res_util_k3,
                             'vl': self.res_util_vlst,
                             'su': self.res_util_soul}}

        split_map = {'cnt': {'drop': ['cnt_cum'],
                             'target': 'cnt_cum'},
                     'disc': {'drop': ['disc_mean'],
                              'target': 'disc_mean'},
                     'util': {
                         'drop': ['util_cum', 'util_rate_cum'],
                         'target': 'util_rate_cum'}}

        return data_map, split_map