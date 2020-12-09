import os
import pickle
import datetime as dt
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor


class WeeklyReport(object):

    REGRESSORS = {"Extra Trees Regressor": ExtraTreesRegressor(),
                  "extr": ExtraTreesRegressor}

    def __init__(self, res_status_last_week: str, res_status_this_week: str, res_status_cancel_this_week: str,
                 res_confirm_last_week: str, disc_confirm_last_week: str, disc_rec_last_week: str,
                 start_day: str, end_day: str, apply_last_week: str, apply_this_week: str):
        # Path
        self.path_hx_data = os.path.join('..', 'result', 'data', 'model_2', 'hx', 'car')
        self.path_model = os.path.join('..', 'result', 'model', 'model_2')
        self.path_sales_per_res = os.path.join('..', 'result', 'data', 'sales_prediction')
        # Date
        self.res_status_last_week = res_status_last_week
        self.res_status_this_week = res_status_this_week
        self.res_status_cancel_this_week = res_status_cancel_this_week
        self.res_confirm_last_week = res_confirm_last_week
        self.disc_confirm_last_week = disc_confirm_last_week
        self.disc_rec_last_week = disc_rec_last_week
        self.apply_this_week = apply_this_week
        self.apply_last_week = apply_last_week
        self.start_day = start_day
        self.end_day = end_day
        # Data
        self.res_re: pd.DataFrame = pd.DataFrame()
        self.cancel_re: pd.DataFrame = pd.DataFrame()
        self.capa_re: dict = {}
        self.disc_re: dict = {}
        self.disc_rec: dict = {}
        self.season_re: dict = {}
        # data type
        self.data_type: list = ['cnt', 'disc', 'util']
        self.car_type: list = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
        self.data_type_map: dict = {'cnt': 'cnt_cum', 'disc': 'cnt_cum', 'util': 'util_cum'}

        # inintialize mapping dictionary
        self.data_hx_map: dict = dict()
        self.split_map: dict = dict()
        self.rent_fee_hx: dict = {}
        self.rent_cdw_hx: dict = {}
        self.day_to_season: dict = {}

        # car grade
        self.grade_1_6 = ['ALL NEW K3 (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)',
                          '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        self.model_type = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
        self.model_type_map = {'av_ad': '아반떼 AD (G) F/L', 'av_new': '올 뉴 아반떼 (G)',
                               'k3': 'ALL NEW K3 (G)', 'soul': '쏘울 부스터 (G)',
                               'vlst': '더 올 뉴 벨로스터 (G)'}
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
        res_last_week, res_this_week, res_cancel_this_week, res_confirmed = self._load_data()

        #
        self._load_fee_hx()
        self.day_to_season = self._get_seasonal_map()
        self.capa_re = self._get_capa_re()
        self.disc_re = self._get_disc_re()
        self.disc_rec = self._get_disc_rec()
        self.season_re = self._get_season_re()

        # Change data type
        res_last_week['rent_day'] = pd.to_datetime(res_last_week['rent_day'], format='%Y-%m-%d')
        res_this_week['rent_day'] = pd.to_datetime(res_this_week['rent_day'], format='%Y-%m-%d')
        res_cancel_this_week['rent_datetime'] = pd.to_datetime(res_cancel_this_week['rent_datetime'],
                                                               format='%Y-%m-%d %H:%M:%S')
        res_cancel_this_week['cancel_datetime'] = pd.to_datetime(res_cancel_this_week['cancel_datetime'],
                                                                 format='%Y-%m-%d %H:%M:%S')
        res_cancel_this_week['rent_day'] = res_cancel_this_week['rent_datetime'].dt.strftime('%Y%m%d')
        res_cancel_this_week['rent_day'] = pd.to_datetime(res_cancel_this_week['rent_day'], format='%Y%m%d')

        # Filter datetime
        last_week_dt = dt.datetime.strptime(self.res_status_last_week, '%y%m%d')
        start_day_dt = dt.datetime.strptime(self.start_day, '%Y%m%d')
        end_day_dt = dt.datetime.strptime(self.end_day, '%Y%m%d')
        apply_last_dt = dt.datetime.strptime(self.apply_last_week, '%y%m%d')
        apply_this_dt = dt.datetime.strptime(self.apply_this_week, '%y%m%d')
        res_last_week = res_last_week[(res_last_week['rent_day'] >= last_week_dt) &
                                      (res_last_week['rent_day'] <= end_day_dt)]
        res_this_week = res_this_week[(res_this_week['rent_day'] >= last_week_dt) &
                                      (res_this_week['rent_day'] <= end_day_dt)]
        res_cancel_this_week = res_cancel_this_week[(res_cancel_this_week['cancel_datetime'] >= apply_last_dt) &
                                                    (res_cancel_this_week['cancel_datetime'] < apply_this_dt)]

        # Filter 1.6 grade cars
        res_last_week = res_last_week[res_last_week['res_model_nm'].isin(self.grade_1_6)]
        res_this_week = res_this_week[res_this_week['res_model_nm'].isin(self.grade_1_6)]
        res_cancel_this_week = res_cancel_this_week[res_cancel_this_week['res_model_nm'].isin(self.grade_1_6)]

        # Filter canceled data
        res_cancel_this_week = res_cancel_this_week[res_cancel_this_week['status'].isin(self.status_cancel)]

        # Group
        res_last_week = self._group_model(df=res_last_week)
        res_this_week = self._group_model(df=res_this_week)
        res_cancel_this_week = self._group_model(df=res_cancel_this_week)

        # Drop Unnecessary columns
        res_last_week = res_last_week.drop(columns=self.drop_col_res, errors='ignore')
        res_this_week = res_this_week.drop(columns=self.drop_col_res, errors='ignore')
        res_cancel_this_week = res_cancel_this_week.drop(columns=self.drop_col_cancel, errors='ignore')

        # Drop Duplicate
        res_last_week = res_last_week.drop_duplicates(subset=['res_num'])
        res_this_week = res_this_week.drop_duplicates(subset=['res_num'])

        # Reservation count
        cnt_last_week = res_last_week.groupby(by=['model', 'rent_day']).count()['res_num']
        cnt_this_week = res_this_week.groupby(by=['model', 'rent_day']).count()['res_num']
        cnt_cancel_this_week = res_cancel_this_week.groupby(by=['rent_day', 'model']).count()['res_num']
        cnt_last_week = cnt_last_week.reset_index(level=(0, 1))
        cnt_this_week = cnt_this_week.reset_index(level=(0, 1))
        cnt_cancel_this_week = cnt_cancel_this_week.reset_index(level=(0, 1))
        cnt_cancel_this_week = cnt_cancel_this_week.rename(columns={'res_num': 'canceled'})

        # Change to utilization rate
        util_last_week = self._get_res_util(df=res_last_week)
        util_this_week = self._get_res_util(df=res_this_week)

        util_last_week_grp = util_last_week.groupby(by=['model', 'rent_day']).sum()['util']
        util_this_week_grp = util_this_week.groupby(by=['model', 'rent_day']).sum()['util']
        util_last_week_grp = util_last_week_grp.reset_index(level=(0, 1))
        util_this_week_grp = util_this_week_grp.reset_index(level=(0, 1))

        util_last_week_grp['rent_mon'] = util_last_week_grp['rent_day'].dt.strftime('%Y%m')
        util_this_week_grp['rent_mon'] = util_this_week_grp['rent_day'].dt.strftime('%Y%m')
        util_last_week_grp['util_rate'] = util_last_week_grp[['rent_mon', 'model', 'util']].apply(self._calc_util_rate,
                                                                                                  axis=1)
        util_this_week_grp['util_rate'] = util_this_week_grp[['rent_mon', 'model', 'util']].apply(self._calc_util_rate,
                                                                                                  axis=1)
        util_last_week_grp = util_last_week_grp.drop(columns=['rent_mon'])
        util_this_week_grp = util_this_week_grp.drop(columns=['rent_mon'])

        # scaling
        util_last_week_grp['util'] = np.round(util_last_week_grp['util'].to_numpy(), 1)
        util_this_week_grp['util'] = np.round(util_this_week_grp['util'].to_numpy(), 1)
        util_last_week_grp['util_rate'] = np.round(util_last_week_grp['util_rate'].to_numpy() * 100, 1)
        util_this_week_grp['util_rate'] = np.round(util_this_week_grp['util_rate'].to_numpy() * 100, 1)

        # Reservation applying discount
        disc_last_week = util_last_week.groupby(by=['model', 'rent_day']).mean()['discount']
        disc_this_week = util_this_week.groupby(by=['model', 'rent_day']).mean()['discount']
        disc_last_week = disc_last_week.reset_index(level=(0, 1))
        disc_this_week = disc_this_week.reset_index(level=(0, 1))

        # Scaling
        disc_last_week['discount'] = np.round(disc_last_week['discount'].to_numpy(), 1)
        disc_this_week['discount'] = np.round(disc_this_week['discount'].to_numpy(), 1)

        # Merge

        result_last_week = pd.merge(util_last_week_grp, disc_last_week, how='left', on=['model', 'rent_day'],
                                    left_index=True, right_index=False)
        result_last_week = pd.merge(result_last_week, cnt_last_week, how='left', on=['model', 'rent_day'],
                                    left_index=True, right_index=False)

        result_this_week = pd.merge(util_this_week_grp, disc_this_week, how='left', on=['model', 'rent_day'],
                                    left_index=True, right_index=False)
        result_this_week = pd.merge(result_this_week, cnt_this_week, how='left', on=['model', 'rent_day'],
                                    left_index=True, right_index=False)

        # Rename columns
        result_last_week = result_last_week.rename(columns={'res_num': 'cnt_last_week', 'util': 'util_last_week',
                                                   'util_rate': 'util_rate_bf', 'discount': 'disc_last_week'})
        result_this_week = result_this_week.rename(columns={'res_num': 'cnt_this_week', 'util': 'util_this_week',
                                                   'util_rate': 'util_rate_af', 'discount': 'disc_this_week'})

        result = pd.merge(result_last_week, result_this_week, how='outer', on=['model', 'rent_day'],
                          left_index=True, right_index=False)
        result = pd.merge(result, cnt_cancel_this_week, how='left', on=['model', 'rent_day'],
                          left_index=True, right_index=False)

        # Filter days
        result = result[(result['rent_day'] >= start_day_dt) & (result['rent_day'] <= end_day_dt)]

        # Fill NA
        result = result.fillna(0)

        # Add Season
        result['season'] = result['rent_day'].apply(lambda x: self.day_to_season[x])

        # Add capacity columns
        result['rent_mon'] = result['rent_day'].dt.strftime('%Y%m')
        result['capa'] = result[['rent_mon', 'model']].apply(self._set_capa, axis=1)
        result['cnt_rate_last_week'] = result['cnt_last_week'] / result['capa']
        result['cnt_rate_this_week'] = result['cnt_this_week'] / result['capa']

        # sales
        result['sales_per_res'] = result[['season', 'model']].apply(self._set_fee, kinds='fee', axis=1)
        result['cdw_fee'] = result[['season', 'model']].apply(self._set_fee, kinds='cdw', axis=1)

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
                 'disc_applied', 'disc_rec', 'disc_diff', 'disc_last_week', 'disc_this_week',
                 'util_this_week', 'util_exp_af', 'util_diff',
                 'canceled', 'exp_sales_chg']

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

        for model in self.model_type:   # av_ad / av_new / k3 / soul / vlst
            date_df = pd.DataFrame({'rent_day': date_range, 'season': season, 'lead_time': lt_af_vec})
            temp = result[result['model'] == model].sort_values(by='rent_day')
            temp = temp.reset_index(drop=True)
            temp = pd.merge(date_df, temp, how='left', on=['rent_day', 'season'], left_index=True, right_index=False)
            temp = temp.fillna(0)

            temp['disc_applied'] = [self.disc_re[rent_day] for rent_day in temp['rent_day']]
            temp['disc_rec'] = [self.disc_rec[self.model_grp[model]][rent_day][3] for rent_day in temp['rent_day']]

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

    def calc_sales(self, res_confirm_day_from: str, res_confirm_day_to: str, apply_last_week: str):
        res_confirm_day_from_dt = dt.datetime.strptime(res_confirm_day_from, '%y%m%d')
        res_confirm_day_to_dt = dt.datetime.strptime(res_confirm_day_to, '%y%m%d')

        # Load dataset
        res_last_week, res_this_week, res_cancel, res_confirmed = self._load_data()

        # Change data type
        res_confirmed['rent_day'] = pd.to_datetime(res_confirmed['rent_day'], format='%Y-%m-%d')
        res_last_week['rent_day'] = pd.to_datetime(res_last_week['rent_day'], format='%Y-%m-%d')
        res_cancel['cancel_datetime'] = pd.to_datetime(res_cancel['cancel_datetime'], format='%Y-%m-%d')

        # Filter datetime
        last_week_dt = dt.datetime.strptime(self.res_status_last_week, '%y%m%d')
        apply_last_week_dt = dt.datetime.strptime(apply_last_week, '%y%m%d')
        end_day_dt = dt.datetime.strptime(self.end_day, '%Y%m%d')
        res_confirmed = res_confirmed[(res_confirmed['rent_day'] >= last_week_dt) &
                                      (res_confirmed['rent_day'] <= end_day_dt)]
        res_cancel = res_cancel[res_cancel['cancel_datetime'] < apply_last_week_dt]


        # Filter 1.6 grade cars
        res_confirmed = res_confirmed[res_confirmed['res_model_nm'].isin(self.grade_1_6)]
        res_last_week = res_last_week[res_last_week['res_model_nm'].isin(self.grade_1_6)]
        res_cancel = res_cancel[res_cancel['res_model_nm'].isin(self.grade_1_6)]

        # Group
        res_confirmed = self._group_model(df=res_confirmed)
        res_last_week = self._group_model(df=res_last_week)
        res_cancel = self._group_model(df=res_cancel)

        # Drop Unnecessary columns
        res_confirmed = res_confirmed.drop(columns=self.drop_col_res, errors='ignore')
        res_last_week = res_last_week.drop(columns=self.drop_col_res, errors='ignore')

        res_last_week_new = pd.merge(res_last_week, res_cancel[['res_num', 'status']], how='left', on=['res_num'],
                                     left_index=True, right_index=False)
        res_last_week_new = res_last_week_new.fillna('reserve')

        res_last_week_new = res_last_week_new[res_last_week_new['status'] == 'reserve']

        tot_sales = res_confirmed.groupby(by=['model', 'rent_day']).sum()['tot_fee']
        tot_sales = tot_sales.reset_index(level=(0, 1))

        tot_sales_last_week = res_last_week_new.groupby(by=['model', 'rent_day']).sum()['tot_fee']
        tot_sales_last_week = tot_sales_last_week.reset_index(level=(0, 1))

        # Filter datetimes
        tot_sales = tot_sales[(tot_sales['rent_day'] >= res_confirm_day_from_dt) &
                              (tot_sales['rent_day'] <= res_confirm_day_to_dt)]

        tot_sales_last_week = tot_sales_last_week[(tot_sales_last_week['rent_day'] >= res_confirm_day_from_dt) &
                                                  (tot_sales_last_week['rent_day'] <= res_confirm_day_to_dt)]

        tot_sales_last_week = tot_sales_last_week.rename(columns={'tot_fee': 'tot_fee_last_week'})

        sales = pd.merge(tot_sales, tot_sales_last_week, how='outer', on=['model', 'rent_day'],
                          left_index=True, right_index=False)

        result = pd.DataFrame()

        for model in self.car_type:
            temp = sales[sales['model'] == model]
            temp = temp.reset_index(drop=True)
            result = pd.concat([result, temp], axis=1, ignore_index=True)

        # Save result
        save_path = os.path.join('..', 'result', 'data', 'weekly_report', self.res_status_this_week)

        result.T.to_csv(os.path.join(save_path, 'weekly_report_sales_confirmed.csv'), header=False)

        # Calculate Expect

    def _set_capa(self, x):
        return self.capa_re[(x[0], x[1])]

    def _set_fee(self, x, kinds: str):
        if kinds == 'cdw':
            return self.rent_cdw_hx[x[0]][x[1]]
        elif kinds == 'fee':
            return self.rent_fee_hx[x[0]][x[1]]

    def _get_pred_input(self, season, lead_time, cnt, disc):
        pred_input = {}
        for data_type in self.data_type:
            if data_type == 'disc':
                pred_input[data_type] = np.array([season, lead_time, cnt]).T
            else:
                pred_input[data_type] = np.array([season, lead_time, disc]).T

        return pred_input

    def _load_data_hx(self):
        data_hx_map = defaultdict(dict)
        for data_type in self.data_type:    # cnt / disc / util
            for model in self.car_type:     # av_ad / av_new / k3 / soul / vlst
                data_type_name = self.data_type_map[data_type]
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
        for data_type in self.data_type:
            io_model = {}
            for model in self.model_type:
                split = self._split_to_input_target(data_type=data_type, model=model)
                io_model[model] = split
            io[data_type] = io_model

        return io

    def _split_to_input_target(self, data_type: str, model: str):
        x = self.data_hx_map[data_type][model].drop(columns=self.split_map[data_type]['drop'])
        y = self.data_hx_map[data_type][model][self.split_map[data_type]['target']]

        return {'x': x, 'y': y}

    def _load_best_params(self, regr: str):
        regr_bests = {}
        for data_type in self.data_type:    # cnt / disc / util
            model_bests = {}
            for model in self.model_type:   # av_ad / av_new / k3 / soul / vlst
                f = open(os.path.join(self.path_model, data_type,
                                      regr + '_params_' + self.model_grp[model] + '.pickle'), 'rb')
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

    def _load_data(self):
        # Load recent reservation dataset
        data_path = os.path.join('..', 'input', 'res_status')
        res_last_week = pd.read_csv(os.path.join(data_path, 'res_' + self.res_status_last_week + '.csv'),
                                    delimiter='\t')
        res_this_week = pd.read_csv(os.path.join(data_path, 'res_' + self.res_status_this_week + '.csv'),
                                    delimiter='\t')
        data_path = os.path.join('..', 'input', 'res_confirm')
        res_confirmed = pd.read_csv(os.path.join(data_path, 'res_confirm_' + self.res_confirm_last_week + '.csv'),
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
        res_last_week = res_last_week.rename(columns=res_remap_cols)
        res_this_week = res_this_week.rename(columns=res_remap_cols)
        res_confirmed = res_confirmed.rename(columns=res_remap_cols)

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

        return res_last_week, res_this_week, cancel_this_week, res_confirmed

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
        disc_re = pd.read_csv(os.path.join(load_path, 'disc_complete_' + self.disc_confirm_last_week + '.csv'),
                              delimiter='\t', dtype={'date': str, 'disc': int})
        disc_re['date'] = pd.to_datetime(disc_re['date'], format='%Y%m%d')
        disc_re = {date: disc for date, disc in zip(disc_re['date'], disc_re['disc'])}

        return disc_re

    def _get_disc_rec(self):
        disc_rec = {}
        load_path = os.path.join('..', 'result', 'data', 'recommend', 'summary', self.disc_rec_last_week, 'original')
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

    @staticmethod
    def _get_season_re():
        # Load recent seasonality dataset
        load_path = os.path.join('..', 'input', 'seasonality')
        ss_curr = pd.read_csv(os.path.join(load_path, 'seasonality_curr.csv'), delimiter='\t')
        ss_curr['date'] = pd.to_datetime(ss_curr['date'], format='%Y%m%d')
        season_map = {day: season for day, season in zip(ss_curr['date'], ss_curr['seasonality'])}

        return season_map

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

    @staticmethod
    def _get_seasonal_map():
        # Load recent seasonality dataset
        load_path = os.path.join('..', 'input', 'seasonality')
        ss_curr = pd.read_csv(os.path.join(load_path, 'seasonality_curr.csv'), delimiter='\t')
        ss_curr['date'] = pd.to_datetime(ss_curr['date'], format='%Y%m%d')
        day_to_season = {day: season for day, season in zip(ss_curr['date'], ss_curr['seasonality'])}

        return day_to_season

    def _load_fee_hx(self):
        sales_per_res = pd.read_csv(os.path.join(self.path_sales_per_res, 'sales_per_res.csv'), encoding='euc-kr')
        rent_fee = defaultdict(dict)
        rent_cdw = defaultdict(dict)
        sales_per_res['res_model'] = sales_per_res['res_model'].apply(lambda x: self.model_type_map_rev[x])
        for season, model, fee, cdw in zip(sales_per_res['seasonality'],
                                           sales_per_res['res_model'],
                                           sales_per_res['rent_fee_org'],
                                           sales_per_res['cdw_fee']):
            rent_fee[season].update({model: int(fee)})
            rent_cdw[season].update({model: int(cdw)})

        self.rent_fee_hx = rent_fee
        self.rent_cdw_hx = rent_cdw
