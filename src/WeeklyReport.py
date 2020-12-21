from Utility import Utility

import os
import pickle
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor


class WeeklyReport(object):

    REGRESSORS = {"Extra Trees Regressor": ExtraTreesRegressor(),
                  "extr": ExtraTreesRegressor}

    def __init__(self, res_status_last_week: str, res_status_this_week: str, res_status_cancel_this_week: str,
                 res_confirm_last_week: str, disc_confirm_last_week: str, disc_rec_last_week: str,
                 disc_rec_2w_ago: str, disc_rec_3w_ago: str, start_day: str, end_day: str, apply_last_week: str,
                 apply_this_week: str, res_confirm_day_from: str, res_confirm_day_to: str, res_confirm_bf: str):
        self.utility = Utility
        # Path
        self.path_hx_data = os.path.join('..', 'result', 'data', 'model_2', 'hx', 'car')
        self.path_model = os.path.join('..', 'result', 'model', 'res_prediction')
        self.path_sales_per_res = os.path.join('..', 'result', 'data', 'sales_prediction')
        # Date
        self.res_status_last_week = res_status_last_week
        self.res_status_this_week = res_status_this_week
        self.res_status_cancel_this_week = res_status_cancel_this_week
        self.res_confirm_last_week = res_confirm_last_week
        self.disc_confirm_last_week = disc_confirm_last_week
        self.disc_rec_last_week = disc_rec_last_week
        self.disc_rec_2w_ago = disc_rec_2w_ago
        self.disc_rec_3w_ago = disc_rec_3w_ago
        self.apply_this_week = apply_this_week
        self.apply_last_week = apply_last_week
        self.start_day = start_day
        self.end_day = end_day
        self.res_confirm_day_from = res_confirm_day_from
        self.res_confirm_day_to = res_confirm_day_to
        self.res_confirm_bf = res_confirm_bf

        # Data
        self.res_re: pd.DataFrame = pd.DataFrame()
        self.cancel_re: pd.DataFrame = pd.DataFrame()
        self.capa_re: dict = {}
        self.season_re: dict = {}
        self.disc_rec: dict = {}
        self.disc_confirm: dict = {}

        # data type
        self.type_data: list = self.utility.TYPE_DATA
        self.type_data_map: dict = self.utility.TYPE_DATA_MAP
        self.type_model: list = self.utility.TYPE_MODEL
        self.model_nm_map: dict = self.utility.MODEL_NAME_MAP
        self.model_nm_map_rev: dict = self.utility.MODEL_NAME_MAP_REV

        # inintialize mapping dictionary
        self.data_hx_map: dict = dict()
        self.split_map: dict = dict()
        self.rent_fee_hx: dict = {}
        self.rent_cdw_hx: dict = {}

        # car grade
        self.grade_1_6 = self.utility.GRADE_1_6
        self.model_type_map_grp = {'av_ad': 'AVANTE', 'av_new': 'AVANTE', 'k3': 'K3',
                                   'vlst': 'VELOSTER', 'soul': 'SOUL'}
        self.model_grp = {'av_ad': 'av', 'av_new': 'av', 'k3': 'k3', 'soul': 'su', 'vlst': 'vl'}
        self.status_cancel = ['취소', 'no show']
        self.drop_col_res = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm', 'car_grd', 'rent_period_day',
                             'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm', 'sale_purpose',
                             'applied_discount', 'discount_rate', 'member_grd', 'sale_purpose', 'car_kind',
                             'res_model_nm']
        self.drop_col_cancel = ['return_datetime', 'res_channel', 'customer_number', 'tot_fee',
                                'res_model_nm', 'status']

    def post_process(self, type_apply: str, time='re'):
        # Load dataset
        res_last_week, res_this_week, res_cancel_this_week, res_confirmed, res_confirmed_bf = self._load_data(time=time)

        # Load rental/cdw fee of history dataset
        self.rent_fee_hx, self.rent_cdw_hx = self._load_fee_hx()
        # Load recent capacity of each model
        self.capa_re = self._get_capacity(time='re', type_apply=type_apply)
        # Load recent seasonal setting
        self.season_re = self._get_season(time='re')
        # Load recommend discount on last week
        self.disc_rec = self._get_disc_rec()
        # Load confirmed discount on last week
        self.disc_confirm = self._get_disc_confirm_last_week()

        # Change data type
        res_last_week['rent_day'] = pd.to_datetime(res_last_week['rent_day'], format='%Y-%m-%d')
        res_this_week['rent_day'] = pd.to_datetime(res_this_week['rent_day'], format='%Y-%m-%d')
        res_cancel_this_week['rent_datetime'] = pd.to_datetime(res_cancel_this_week['rent_datetime'],
                                                               format='%Y-%m-%d %H:%M:%S')
        res_cancel_this_week['cancel_datetime'] = pd.to_datetime(res_cancel_this_week['cancel_datetime'],
                                                                 format='%Y-%m-%d %H:%M:%S')
        res_cancel_this_week['rent_day'] = res_cancel_this_week['rent_datetime'].dt.strftime('%Y%m%d')
        res_cancel_this_week['rent_day'] = pd.to_datetime(res_cancel_this_week['rent_day'], format='%Y%m%d')

        res_confirmed['rent_day'] = pd.to_datetime(res_confirmed['rent_day'], format='%Y-%m-%d')
        res_confirmed_bf['rent_day'] = pd.to_datetime(res_confirmed_bf['rent_day'], format='%Y-%m-%d')

        # Filter datetime
        res_last_week, res_this_week, res_cancel_this_week = self._filter_date_res(last_week=res_last_week,
                                                                                   this_week=res_this_week,
                                                                                   cc_this_week=res_cancel_this_week)
        res_confirmed, res_confirmed_bf = self._filter_date_res_confirm(res_confirmed=res_confirmed,
                                                                        res_confirmed_bf=res_confirmed_bf)

        # merge
        res_last_week = pd.concat([res_confirmed_bf, res_last_week], axis=0)
        res_this_week = pd.concat([res_confirmed, res_this_week], axis=0)

        # Filter 1.6 grade cars
        res_last_week = self.utility.filter_model_grade(df=res_last_week)
        res_this_week = self.utility.filter_model_grade(df=res_this_week)
        res_cancel_this_week = self.utility.filter_model_grade(df=res_cancel_this_week)

        # Filter canceled data
        res_cancel_this_week = res_cancel_this_week[res_cancel_this_week['status'].isin(self.status_cancel)]

        # Group
        res_last_week = self.utility.cluster_model(df=res_last_week, type_apply=type_apply)
        res_this_week = self.utility.cluster_model(df=res_this_week, type_apply=type_apply)
        res_cancel_this_week = self.utility.cluster_model(df=res_cancel_this_week, type_apply=type_apply)

        # Drop Unnecessary columns
        res_last_week = res_last_week.drop(columns=self.drop_col_res, errors='ignore')
        res_this_week = res_this_week.drop(columns=self.drop_col_res, errors='ignore')
        res_cancel_this_week = res_cancel_this_week.drop(columns=self.drop_col_cancel, errors='ignore')

        # Drop Duplicate
        res_last_week = res_last_week.drop_duplicates(subset=['res_num'])
        res_this_week = res_this_week.drop_duplicates(subset=['res_num'])

        # Reservation count
        cnt_last_week, cnt_this_week, cnt_cancel_this_week = self._group_res_cnt(last_week=res_last_week,
                                                                                 this_week=res_this_week,
                                                                                 cancel_this_week=res_cancel_this_week)

        # Change to utilization rate
        util_last_week = self.utility.get_res_util(df=res_last_week)
        util_this_week = self.utility.get_res_util(df=res_this_week)

        util_last_week_grp = util_last_week.groupby(by=['res_model', 'rent_day']).sum()['util']
        util_this_week_grp = util_this_week.groupby(by=['res_model', 'rent_day']).sum()['util']
        util_last_week_grp = util_last_week_grp.reset_index(level=(0, 1))
        util_this_week_grp = util_this_week_grp.reset_index(level=(0, 1))

        util_last_week_grp['rent_mon'] = util_last_week_grp['rent_day'].dt.strftime('%Y%m')
        util_this_week_grp['rent_mon'] = util_this_week_grp['rent_day'].dt.strftime('%Y%m')
        util_last_week_grp['util_rate'] = util_last_week_grp[['rent_mon', 'res_model',
                                                              'util']].apply(self._calc_util_rate, axis=1)
        util_this_week_grp['util_rate'] = util_this_week_grp[['rent_mon', 'res_model',
                                                              'util']].apply(self._calc_util_rate, axis=1)
        util_last_week_grp = util_last_week_grp.drop(columns=['rent_mon'])
        util_this_week_grp = util_this_week_grp.drop(columns=['rent_mon'])

        # scaling
        util_last_week_grp['util'] = np.round(util_last_week_grp['util'].to_numpy(), 1)
        util_this_week_grp['util'] = np.round(util_this_week_grp['util'].to_numpy(), 1)
        util_last_week_grp['util_rate'] = np.round(util_last_week_grp['util_rate'].to_numpy() * 100, 1)
        util_this_week_grp['util_rate'] = np.round(util_this_week_grp['util_rate'].to_numpy() * 100, 1)

        # Reservation applying discount
        disc_last_week = util_last_week.groupby(by=['res_model', 'rent_day']).mean()['discount']
        disc_this_week = util_this_week.groupby(by=['res_model', 'rent_day']).mean()['discount']
        disc_last_week = disc_last_week.reset_index(level=(0, 1))
        disc_this_week = disc_this_week.reset_index(level=(0, 1))

        # Scaling
        disc_last_week['discount'] = np.round(disc_last_week['discount'].to_numpy(), 1)
        disc_this_week['discount'] = np.round(disc_this_week['discount'].to_numpy(), 1)

        # Merge

        result_last_week = pd.merge(util_last_week_grp, disc_last_week, how='left', on=['res_model', 'rent_day'],
                                    left_index=True, right_index=False)
        result_last_week = pd.merge(result_last_week, cnt_last_week, how='left', on=['res_model', 'rent_day'],
                                    left_index=True, right_index=False)

        result_this_week = pd.merge(util_this_week_grp, disc_this_week, how='left', on=['res_model', 'rent_day'],
                                    left_index=True, right_index=False)
        result_this_week = pd.merge(result_this_week, cnt_this_week, how='left', on=['res_model', 'rent_day'],
                                    left_index=True, right_index=False)

        # Rename columns
        result_last_week = result_last_week.rename(columns={'res_num': 'cnt_last_week', 'util': 'util_last_week',
                                                   'util_rate': 'util_rate_bf', 'discount': 'disc_last_week'})
        result_this_week = result_this_week.rename(columns={'res_num': 'cnt_this_week', 'util': 'util_this_week',
                                                   'util_rate': 'util_rate_af', 'discount': 'disc_this_week'})

        result = pd.merge(result_last_week, result_this_week, how='outer', on=['res_model', 'rent_day'],
                          left_index=True, right_index=False)
        result = pd.merge(result, cnt_cancel_this_week, how='left', on=['res_model', 'rent_day'],
                          left_index=True, right_index=False)

        # Filter days
        start_day_dt = dt.datetime.strptime(self.start_day, '%Y%m%d')
        end_day_dt = dt.datetime.strptime(self.end_day, '%Y%m%d')
        result = result[(result['rent_day'] >= start_day_dt) & (result['rent_day'] <= end_day_dt)]

        # Fill NA
        result = result.fillna(0)

        # Add Season
        result['season'] = result['rent_day'].apply(lambda x: self.season_re[x])

        # Add capacity columns
        result['rent_mon'] = result['rent_day'].dt.strftime('%Y%m')
        result['capa'] = result[['rent_mon', 'res_model']].apply(self._set_capa, axis=1)
        result['cnt_rate_last_week'] = result['cnt_last_week'] / result['capa']
        result['cnt_rate_this_week'] = result['cnt_this_week'] / result['capa']

        # sales
        result['sales_per_res'] = result[['season', 'res_model']].apply(self._set_fee, kinds='fee', axis=1)
        result['cdw_fee'] = result[['season', 'res_model']].apply(self._set_fee, kinds='cdw', axis=1)

        # Calculate season and lead time
        result_dict = {}
        date_range = pd.date_range(start=self.start_day, end=self.end_day, freq='D')
        season = [self.season_re[date] for date in date_range]
        lt_af = list(map(int, (date_range - start_day_dt) / np.timedelta64(1, 'D')))
        lt_bf = [lt + 7 for lt in lt_af]
        lt_to_lt_vec = {-1 * i: (((i // 7) + 24) * -1 if i > 28 else i * -1) for i in range(0, lt_bf[-1] + 1, 1)}
        lt_af_vec = [lt_to_lt_vec[lt * -1] for lt in lt_af]
        lt_bf_vec = [lt_to_lt_vec[lt * -1] for lt in lt_bf]

        rearr = ['rent_day',
                 'cnt_last_week', 'cnt_this_week', 'cnt_chg', 'cnt_exp_bf', 'cnt_exp_af', 'cnt_exp_chg', 'cnt_chg_diff',
                 'canceled', 'disc_applied', 'disc_rec', 'disc_diff', 'disc_last_week', 'disc_this_week',
                 'util_this_week', 'util_exp_af', 'util_diff', 'exp_sales_chg']

        # Load best hyper-parameters
        self.data_hx_map = self._load_data_hx()

        # Define input and output
        self.split_map = self._set_split_map()

        # Split into input and output
        io = self._split_input_target_all()

        # Load best hyper-parameters
        extr_bests = self._load_best_params(regr='extr')

        # fit the model
        fitted = self._fit_model(dataset=io, regr='extr', params=extr_bests)

        for model in self.type_model:   # av_ad / av_new / k3 / soul / vlst
            date_df = pd.DataFrame({'rent_day': date_range, 'season': season, 'lead_time': lt_af_vec})
            temp = result[result['res_model'] == model].sort_values(by='rent_day')
            temp = temp.reset_index(drop=True)
            temp = pd.merge(date_df, temp, how='left', on=['rent_day', 'season'], left_index=True, right_index=False)
            temp = temp.fillna(0)

            disc_confirm_model = self.disc_confirm[model]
            temp['disc_applied'] = [disc_confirm_model[rent_day] for rent_day in temp['rent_day']]
            temp['disc_rec'] = [self.disc_rec[model][rent_day][3] for rent_day in temp['rent_day']]

            temp['cnt_chg'] = temp['cnt_this_week'] - temp['cnt_last_week']
            temp['disc_diff'] = temp['disc_applied'] - temp['disc_rec']

            fitted_model = fitted[model]

            pred_input_bf = self._get_pred_input(season=season, lead_time=lt_bf_vec,
                                                 cnt=temp['cnt_rate_last_week'].to_numpy(),
                                                 disc=temp['disc_rec'].to_numpy())
            pred_input_af = self._get_pred_input(season=season, lead_time=lt_af_vec,
                                                 cnt=temp['cnt_rate_this_week'].to_numpy(),
                                                 disc=temp['disc_rec'].to_numpy())
            for type_key, type_val in fitted_model.items():
                if type_key == 'cnt':
                    temp[type_key + '_exp_bf'] = np.round(type_val.predict(pred_input_bf[type_key]) *
                                                          temp['capa'].to_numpy(), 1)
                    temp[type_key + '_exp_af'] = np.round(type_val.predict(pred_input_af[type_key]) *
                                                          temp['capa'].to_numpy(), 1)
                else:
                    temp[type_key + '_exp_bf'] = np.round(type_val.predict(pred_input_bf[type_key]), 1)
                    temp[type_key + '_exp_af'] = np.round(type_val.predict(pred_input_af[type_key]), 1)

            # Calculate the difference : count / utilization / utilization rate
            temp['cnt_exp_chg'] = temp['cnt_exp_af'] - temp['cnt_exp_bf']
            temp['util_diff'] = temp['util_this_week'] - temp['util_exp_af']
            temp['cnt_chg_diff'] = temp['cnt_chg'] - temp['cnt_exp_chg']

            # Calculate Expected sales
            temp['exp_sales_chg'] = temp['cnt_exp_chg'] * (temp['sales_per_res'] * (1-temp['disc_rec'] / 100) +
                                                           temp['cdw_fee'])

            temp['exp_sales_chg'] = np.round(temp['exp_sales_chg'].to_numpy(), 0)

            # Re-arrange columns
            temp = temp[rearr]
            result_dict[model] = temp

        save_path = os.path.join('..', 'result', 'data', 'weekly_report', self.res_status_this_week)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for model, df in result_dict.items():
            df['rent_day'] = df['rent_day'].dt.strftime('%Y-%m-%d')
            df.T.to_csv(os.path.join(save_path, 'weekly_report_' + model + '.csv'), header=False)

        print("")

    @staticmethod
    def _group_res_cnt(last_week: pd.DataFrame, this_week: pd.DataFrame,
                       cancel_this_week: pd.DataFrame):
        cnt_last_week = last_week.groupby(by=['res_model', 'rent_day']).count()['res_num']
        cnt_this_week = this_week.groupby(by=['res_model', 'rent_day']).count()['res_num']
        cnt_cancel_this_week = cancel_this_week.groupby(by=['rent_day', 'res_model']).count()['res_num']
        cnt_last_week = cnt_last_week.reset_index(level=(0, 1))
        cnt_this_week = cnt_this_week.reset_index(level=(0, 1))
        cnt_cancel_this_week = cnt_cancel_this_week.reset_index(level=(0, 1))
        cnt_cancel_this_week = cnt_cancel_this_week.rename(columns={'res_num': 'canceled'})

        return cnt_last_week, cnt_this_week, cnt_cancel_this_week

    def calc_sales(self, res_confirm_day_from: str, res_confirm_day_to: str, apply_last_week: str,
                   type_apply: str):
        res_confirm_day_from_dt = dt.datetime.strptime(res_confirm_day_from, '%y%m%d')
        res_confirm_day_to_dt = dt.datetime.strptime(res_confirm_day_to, '%y%m%d')

        # Load dataset
        res_last_week, res_this_week, cancel_this_week, res_confirmed, res_confirmed_bf = self._load_data(time='re')

        # Change data type
        res_confirmed['rent_day'] = pd.to_datetime(res_confirmed['rent_day'], format='%Y-%m-%d')
        res_confirmed['res_day'] = pd.to_datetime(res_confirmed['res_day'], format='%Y-%m-%d')

        # Filter 1.6 grade cars
        res_confirmed = res_confirmed[res_confirmed['res_model_nm'].isin(self.grade_1_6)]

        # Group
        res_confirmed = self.utility.cluster_model(df=res_confirmed, type_apply=type_apply)

        # Drop Unnecessary columns
        res_confirmed = res_confirmed.drop(columns=self.drop_col_res, errors='ignore')

        # Filter datetime
        apply_last_week_dt = dt.datetime.strptime('201127', '%y%m%d')
        res_confirmed = res_confirmed[(res_confirmed['rent_day'] >= res_confirm_day_from_dt) &
                                      (res_confirmed['rent_day'] <= res_confirm_day_to_dt)]
        tot_sales = res_confirmed.groupby(by=['res_model', 'rent_day']).sum()['tot_fee']
        tot_sales = tot_sales.reset_index(level=(0, 1))

        res_confirmed_last_week = res_confirmed[res_confirmed['res_day'] <= apply_last_week_dt]
        tot_sales_last_week = res_confirmed_last_week.groupby(by=['res_model', 'rent_day']).sum()['tot_fee']
        tot_sales_last_week = tot_sales_last_week.reset_index(level=(0, 1))

        # Filter datetimes
        tot_sales = tot_sales[(tot_sales['rent_day'] >= res_confirm_day_from_dt) &
                              (tot_sales['rent_day'] <= res_confirm_day_to_dt)]

        tot_sales_last_week = tot_sales_last_week[(tot_sales_last_week['rent_day'] >= res_confirm_day_from_dt) &
                                                  (tot_sales_last_week['rent_day'] <= res_confirm_day_to_dt)]

        tot_sales_last_week = tot_sales_last_week.rename(columns={'tot_fee': 'tot_fee_last_week'})

        sales = pd.merge(tot_sales, tot_sales_last_week, how='outer', on=['res_model', 'rent_day'],
                         left_index=True, right_index=False)

        # Save result
        save_path = os.path.join('..', 'result', 'data', 'weekly_report', self.res_status_this_week)
        sales.T.to_csv(os.path.join(save_path, 'weekly_report_sales_confirmed.csv'), header=False)

        # Calculate Expect

    def _filter_date_res(self, last_week: pd.DataFrame, this_week: pd.DataFrame,
                         cc_this_week: pd.DataFrame):
        rest_confirm_day_from_dt = dt.datetime.strptime(self.res_confirm_day_from, '%y%m%d')
        rest_confirm_day_to_dt = dt.datetime.strptime(self.res_confirm_day_to, '%y%m%d')
        end_day_dt = dt.datetime.strptime(self.end_day, '%Y%m%d')
        apply_last_dt = dt.datetime.strptime(self.apply_last_week, '%y%m%d')
        apply_this_dt = dt.datetime.strptime(self.apply_this_week, '%y%m%d')
        last_week = last_week[(last_week['rent_day'] > rest_confirm_day_to_dt) &
                              (last_week['rent_day'] <= end_day_dt)]
        this_week = this_week[(this_week['rent_day'] > rest_confirm_day_to_dt) &
                              (this_week['rent_day'] <= end_day_dt)]
        cc_this_week = cc_this_week[(cc_this_week['cancel_datetime'] >= apply_last_dt) &
                                    (cc_this_week['cancel_datetime'] < apply_this_dt)]

        return last_week, this_week, cc_this_week

    def _filter_date_res_confirm(self, res_confirmed: pd.DataFrame, res_confirmed_bf: pd.DataFrame):
        rest_confirm_day_from_dt = dt.datetime.strptime(self.res_confirm_day_from, '%y%m%d')
        rest_confirm_day_to_dt = dt.datetime.strptime(self.res_confirm_day_to, '%y%m%d')
        res_confirmed = res_confirmed[res_confirmed['rent_day'] <= rest_confirm_day_to_dt]
        # res_confirmed = res_confirmed[(res_confirmed['rent_day'] >= rest_confirm_day_from_dt) &
        #                               (res_confirmed['rent_day'] <= rest_confirm_day_to_dt)]
        res_confirmed_bf = res_confirmed_bf[res_confirmed_bf['rent_day'] <= rest_confirm_day_to_dt]
        # res_confirmed_bf = res_confirmed_bf[(res_confirmed_bf['rent_day'] >= rest_confirm_day_from_dt) &
        #                               (res_confirmed_bf['rent_day'] <= rest_confirm_day_to_dt)]

        return res_confirmed, res_confirmed_bf

    def _set_capa(self, x):
        return self.capa_re[(x[0], x[1])]

    def _set_fee(self, x, kinds: str):
        if kinds == 'cdw':
            return self.rent_cdw_hx[x[0]][x[1]]
        elif kinds == 'fee':
            return self.rent_fee_hx[x[0]][x[1]]

    def _get_pred_input(self, season, lead_time, cnt, disc):
        pred_input = {}
        for data_type in self.type_data:
            if data_type == 'disc':
                pred_input[data_type] = np.array([season, lead_time, cnt]).T
            else:
                pred_input[data_type] = np.array([season, lead_time, disc]).T

        return pred_input

    def _load_data_hx(self):
        data_hx_map = defaultdict(dict)
        for data_type in self.type_data:    # cnt / disc / util
            for model in self.type_model:     # av_ad / av_new / k3 / soul / vlst
                data_type_name = self.type_data_map[data_type]
                data_hx_map[data_type].update({model: pd.read_csv(os.path.join(self.path_hx_data,
                                               data_type_name, data_type_name + '_' + model + '.csv'))})

        return data_hx_map

    @staticmethod
    def _set_split_map():
        split_map = {'cnt': {'drop': ['cnt_cum'],
                             'target': 'cnt_cum'},
                     'disc': {'drop': ['disc_mean'],
                              'target': 'disc_mean'},
                     'util': {'drop': ['util_cum', 'util_rate_cum'],
                              'target': 'util_cum'},
                     'util_rate': {'drop': ['util_cum', 'util_rate_cum'],
                                   'target': 'util_rate_cum'}}

        return split_map

    def _split_input_target_all(self):
        io = {}
        for data_type in self.type_data:
            io_model = {}
            for model in self.type_model:
                split = self._split_to_input_target(data_type=data_type, model=model)
                io_model[model] = split
            io[data_type] = io_model

        return io

    def _split_to_input_target(self, data_type: str, model: str):
        x = self.data_hx_map[data_type][model].drop(columns=self.split_map[data_type]['drop'])
        y = self.data_hx_map[data_type][model][self.split_map[data_type]['target']]

        return {'x': x, 'y': y}

    def _load_best_params(self, regr: str, model_detail='car'):
        regr_bests = {}
        for data_type in self.type_data:    # cnt / disc / util
            model_bests = {}
            for model in self.type_model:   # av_ad / av_new / k3 / soul / vlst
                f = open(os.path.join(self.path_model, model_detail, data_type,
                                      regr + '_params_' + model + '.pickle'), 'rb')
                model_bests[model] = pickle.load(f)
                f.close()
            regr_bests[data_type] = model_bests

        return regr_bests

    def _fit_model(self, dataset: dict, regr: str, params: dict):
        fitted = defaultdict(dict)
        for type_key, type_val in params.items():
            for model_key, model_val in type_val.items():
                model = self.REGRESSORS[regr](**model_val)
                data = dataset[type_key][model_key]
                model.fit(data['x'], data['y'])
                fitted[model_key].update({type_key: model})

        return fitted

    def _calc_util_rate(self, x):
        return x[2] / self.capa_re[(x[0], x[1])]

    def _load_data(self, time: str):
        # Load recent reservation dataset
        res_last_week = self.utility.load_res(time=time, status_update_day=self.res_status_last_week)
        res_this_week = self.utility.load_res(time=time, status_update_day=self.res_status_this_week)
        data_path = os.path.join('..', 'input', 'res_confirm')
        res_confirmed = pd.read_csv(os.path.join(data_path, 'res_confirm_' + self.res_confirm_last_week + '.csv'),
                                    delimiter='\t')
        res_confirmed_bf = pd.read_csv(os.path.join('..', 'input', 'res_status',
                                                    'res_status_' + self.res_confirm_bf + '.csv'), delimiter='\t')

        # Rename columns
        res_last_week = res_last_week.rename(columns=self.utility.RENAME_COL_RES)
        res_this_week = res_this_week.rename(columns=self.utility.RENAME_COL_RES)
        res_confirmed = res_confirmed.rename(columns=self.utility.RENAME_COL_RES)
        res_confirmed_bf = res_confirmed_bf.rename(columns=self.utility.RENAME_COL_RES)

        # Data Preprocessing
        res_confirmed['res_route'] = 0    # Exception
        res_confirmed = res_confirmed[res_confirmed['sale_purpose'] == '단기']
        res_this_week = pd.concat([res_this_week, res_confirmed], axis=0)

        # Load recent reservation cancel dataset
        data_path = os.path.join('..', 'input', 'cancel')
        cancel_this_week = pd.read_csv(os.path.join(data_path, 'res_cancel_' +
                                                    self.res_status_cancel_this_week + '.csv'), delimiter='\t')

        # Rename columns
        cancel_remap_cols = {
            '상태': 'status', '계약번호': 'res_num', '대여차종': 'res_model_nm',
            '대여일시': 'rent_datetime', '반납일시': 'return_datetime', '예약채널': 'res_channel',
            '고객번호': 'customer_number', '총청구액': 'tot_fee', '최종수정일시': 'cancel_datetime'}
        cancel_this_week = cancel_this_week.rename(columns=cancel_remap_cols)

        return res_last_week, res_this_week, cancel_this_week, res_confirmed, res_confirmed_bf

    def _get_capacity(self, time: str, type_apply: str):
        # Initial capacity of each model
        capacity = self.utility.load_capacity(time=time, type_apply=type_apply)
        capa_map = self.utility.make_capa_map(df=capacity)

        return capa_map

    def _get_disc_confirm_last_week(self):
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'disc_confirm')
        disc_comfirm = pd.read_csv(os.path.join(load_path, 'disc_confirm_' + self.disc_confirm_last_week + '.csv'),
                                   delimiter='\t', dtype={'date': str, 'disc': int})
        disc_comfirm['date'] = pd.to_datetime(disc_comfirm['date'], format='%Y%m%d')
        disc_comfirm_dict = defaultdict(dict)
        for date, model, disc in zip(disc_comfirm['date'], disc_comfirm['model'], disc_comfirm['disc']):
            disc_comfirm_dict[self.model_nm_map[model]].update({date: disc})

        return disc_comfirm_dict

    def _get_disc_rec(self):
        load_path_last_week = os.path.join('..', 'result', 'data', 'recommend', 'summary',
                                           self.disc_rec_last_week, 'original')
        load_path_2w_ago = os.path.join('..', 'result', 'data', 'recommend', 'summary',
                                        self.disc_rec_2w_ago, 'original')
        load_path_3w_ago = os.path.join('..', 'result', 'data', 'recommend', 'summary',
                                        self.disc_rec_3w_ago, 'original')

        filter_cols = ['대여일', '기대 예약건수', '기대 가동대수(시간)', '기대가동률', '추천 할인율']
        rename_cols = {'대여일': 'rent_day', '기대 예약건수': 'exp_cnt', '기대 가동대수(시간)': 'exp_util',
                       '기대가동률': 'exp_util_rate', '추천 할인율': 'disc_rec'}
        rec_last_week = {}
        for model_grp in self.type_model:
            disc_rec_last_week = dt.datetime.strptime(self.disc_rec_last_week, '%Y%m%d')
            disc_rec_2w_ago = dt.datetime.strptime(self.disc_rec_2w_ago, '%Y%m%d')

            lt_week = pd.read_csv(os.path.join(load_path_last_week, 'rec_summary_' + model_grp + '.csv'),
                                  encoding='euc-kr')
            lt_week = lt_week[filter_cols]
            lt_week = lt_week.rename(columns=rename_cols)
            lt_week['rent_day'] = pd.to_datetime(lt_week['rent_day'], format='%Y-%m-%d')

            week_2_ago = pd.read_csv(os.path.join(load_path_2w_ago, 'rec_summary_' + model_grp + '.csv'),
                                       encoding='euc-kr')
            week_2_ago = week_2_ago[filter_cols]
            week_2_ago = week_2_ago.rename(columns=rename_cols)
            week_2_ago['rent_day'] = pd.to_datetime(week_2_ago['rent_day'], format='%Y-%m-%d')
            week_2_ago = week_2_ago[week_2_ago['rent_day'] < disc_rec_last_week]

            week_3_ago = pd.read_csv(os.path.join(load_path_3w_ago, 'rec_summary_' + model_grp + '.csv'),
                                       encoding='euc-kr')
            week_3_ago = week_3_ago[filter_cols]
            week_3_ago = week_3_ago.rename(columns=rename_cols)
            week_3_ago['rent_day'] = pd.to_datetime(week_3_ago['rent_day'], format='%Y-%m-%d')
            week_3_ago = week_3_ago[week_3_ago['rent_day'] < disc_rec_2w_ago]

            lt_week = pd.concat([week_3_ago, week_2_ago, lt_week], axis=0)

            rec_last_week[model_grp] = {date: [cnt, util, util_rate, disc] for date, cnt,
                                                                               util, util_rate, disc in zip(lt_week['rent_day'], lt_week['exp_cnt'],
                                                                                                            lt_week['exp_util'], lt_week['exp_util_rate'], lt_week['disc_rec'])}
        return rec_last_week

    def _get_season(self, time: str):
        # Load recent seasonality dataset
        season_re = self.utility.load_season(time=time)
        season_map = {day: season for day, season in zip(season_re['rent_day'], season_re['seasonality'])}

        return season_map

    def _load_fee_hx(self):
        sales_per_res = pd.read_csv(os.path.join(self.path_sales_per_res, 'sales_per_res.csv'), encoding='euc-kr')
        rent_fee = defaultdict(dict)
        rent_cdw = defaultdict(dict)
        for season, model, fee, cdw in zip(sales_per_res['seasonality'], sales_per_res['res_model'],
                                           sales_per_res['rent_fee_org'], sales_per_res['cdw_fee']):
            rent_fee[season].update({model: int(fee)})
            rent_cdw[season].update({model: int(cdw)})

        return rent_fee, rent_cdw
