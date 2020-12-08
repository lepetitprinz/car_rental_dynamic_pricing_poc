import os
import copy
import pickle
import datetime as dt
from datetime import timedelta
from collections import defaultdict
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor

rc('font', family='NanumBarunGothic')


class SalesPredict(object):

    REGRESSORS = {"Extra Trees Regressor": ExtraTreesRegressor(),
                  "extr": ExtraTreesRegressor}

    def __init__(self, res_update_day: str):
        # Path of data & model
        self.path_input = os.path.join('..', 'input')
        self.path_trend_hx = os.path.join('..', 'result', 'data', 'model_2', 'hx', 'car')
        self.path_sales_per_res = os.path.join('..', 'result', 'data', 'sales_prediction')
        self.path_model = os.path.join('..', 'result', 'model', 'res_pred_lead_time')

        # Data & model types
        self.data_type = ['cnt_inc', 'cnt_cum', 'util_inc', 'util_cum', 'disc']
        self.model_type = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
        self.model_1_6 = ['ALL NEW K3 (G)', '아반떼 AD (G)', '아반떼 AD (G) F/L',
                          '올 뉴 아반떼 (G)', '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']
        self.model_type_map = {'av_ad': '아반떼 AD (G) F/L', 'av_new': '올 뉴 아반떼 (G)',
                               'k3': 'ALL NEW K3 (G)', 'soul': '쏘울 부스터 (G)',
                               'vlst': '더 올 뉴 벨로스터 (G)'}
        self.model_type_map_rev = {'아반떼 AD (G) F/L': 'av_ad', '올 뉴 아반떼 (G)': 'av_new',
                                   'ALL NEW K3 (G)': 'k3', '쏘울 부스터 (G)': 'soul',
                                   '더 올 뉴 벨로스터 (G)': 'vlst'}

        # Reservation fee
        self.fee_group = {'아반떼 AD (G) F/L': 'm1', 'ALL NEW K3 (G)': 'm1', '더 올 뉴 벨로스터 (G)': 'm1',
                          '올 뉴 아반떼 (G)': 'm2', '쏘울 부스터 (G)': 'm2'}

        self.fee_per_times = {'fee': {'m1': {1: 17000, 2: 34000, 3: 51000, 4: 67000, 5: 67000, 6: 67000, 7: 84000,
                                             8: 96000, 9: 96000, 10: 96000, 11: 96000, 12: 96000, 13: 113000,
                                             14: 120000, 15: 120000, 16: 120000, 17: 120000, 18: 120000, 19: 120000,
                                             20: 120000, 21: 120000, 22: 120000, 23: 120000, 24: 120000},
                                      'm2': {1: 18000, 2: 36000, 3: 54000, 4: 73000, 5: 73000, 6: 73000, 7: 91000,
                                             8: 104000, 9: 104000, 10: 104000, 11: 104000, 12: 104000, 13: 122000,
                                             14: 130000, 15: 130000, 16: 130000, 17: 130000, 18: 130000, 19: 130000,
                                             20: 130000, 21: 130000, 22: 130000, 23: 130000, 24: 130000}},
                              'cdw': {'m1': {1: 20000, 2: 40000, 3: 54000, 4: 72000, 5: 90000, 6: 108000,  7: 112000,
                                             8: 126000, 9: 126000, 10: 140000, 11: 154000,  12: 168000, 13: 182000,
                                             14: 196000, 15: 90000, 16: 96000, 17: 102000},
                                      'm2': {1: 24000, 2: 48000, 3: 66900, 4: 89200, 5: 111500, 6: 133800, 7: 119700,
                                             8: 136800, 9: 153900, 10: 171000, 11: 188100, 12: 205200, 13: 222300,
                                             14: 239400, 15: 91500, 16: 97600, 17: 103700}}}

        # Initial Setting
        self.random_state = 2020    # Data split randomness
        self.test_size = 0.2        # Test dataset size

        # Dataset
        self.res_data_hx: dict = {}
        self.rent_fee_hx: dict = {}
        self.rent_cdw_hx: dict = {}

        # Initialize mapping dictionary
        self.data_map: dict = dict()
        self.split_map: dict = dict()
        self.param_grids = dict()

        # Prediction variables
        # Initial values of variables
        self.res_update_day = res_update_day
        self.res_cnt_init: dict = {}
        self.res_util_init: dict = {}
        self.res_sales_init: dict = {}
        self.day_to_season: dict = {}
        self.disc_re: dict = {}
        self.capa_re: dict = {}
        self.avg_unavail_capa = 2

        # Lead time
        self.lt: np.array = []
        self.lt_vec: np.array = []
        self.lt_to_lt_vec: dict = {}

    def data_preprocessing(self):
        # Load reservation history dataset
        res_hx = self._load_data_hx()

        # Filter 1.6 grade models
        res_hx = res_hx[res_hx['res_model_nm'].isin(self.model_1_6)]
        res_hx = res_hx.reset_index(drop=True)

        res_hx = self._cluster_by_group(df=res_hx)

        # Calculate expected reservation periods
        res_hx = self._calc_exp_res_period(df=res_hx)

        # Filter date
        res_hx = res_hx[res_hx['rent_datetime'] >= dt.datetime(2018, 1, 1)]
        res_hx = res_hx.reset_index(drop=True)

        res_hx = res_hx.rename(columns={'car_rent_fee': 'rent_fee'})

        # Group by season & model
        # res_hx_grp = res_hx.groupby(by=['seasonality', 'res_model']).mean()['rent_period_hours']
        res_hx_grp = res_hx.groupby(by=['seasonality', 'res_model']).mean()
        res_hx_grp['rent_fee_org'] = res_hx_grp['rent_fee'] / (1 - res_hx_grp['discount'] / 100)
        res_hx_grp = res_hx_grp.reset_index(level=(0, 1))

        res_hx_grp['rent_fee_org'] = np.round(res_hx_grp['rent_fee_org'].to_numpy(), 0)
        res_hx_grp['cdw_fee'] = np.round(res_hx_grp['cdw_fee'].to_numpy(), 0)

        # Calculate sales by each model
        # res_hx_grp = self._calc_exp_sales(df=res_hx_grp)

        res_hx_grp = res_hx_grp.drop(columns=['rent_fee', 'tot_fee'],
                                     errors='ignore')

        res_hx_grp.to_csv(os.path.join('..', 'result', 'data', 'sales_prediction', 'sales_per_res.csv'),
                          index=False, encoding='euc-kr')

    def train(self):
        # Load dataset
        self._load_hx()

        # Define input and output
        self._set_split_map()

        # Split into input and output
        m2_io = self._split_input_target_all()

        # Split dataset into train and test dataset
        m2 = self._split_train_test_all(data=m2_io)

        # Update parameter grids
        self.param_grids.update({"Extra Trees Regressor": {
            'n_estimators': list(np.arange(100, 500, 100)),
            'criterion': ['mse'],
            'min_samples_split': list(np.arange(2, 6, 1)),  # minimum number of samples required to split inner node
            'min_samples_leaf': list(np.arange(1, 6, 1)),  # have the effect of smoothing the model
            'max_features': ['auto']}})

        extr_bests = self._grid_search_cross_validation(data=m2, regr='Extra Trees Regressor')

        # Save best parameter grid
        self._save_best_params(regr='extr', regr_bests=extr_bests)

        print('Training finished')

    def predict(self, pred_days: list, apply_day: str):
        # Load dataset
        self._load_hx()

        # Set initial variables
        self._set_pred_init_variables()

        # Set recent data mapping
        self.res_cnt_init, self.res_util_init, self.res_sales_init = self._get_init_res_cnt()

        # Define input and output
        self._set_split_map()

        # Split into input and output
        m2_io = self._split_input_target_all()

        # Load best hyper-parameters
        extr_bests = self._load_best_params(regr='extr')

        # fit the model
        fitted = self._fit_model(dataset=m2_io, regr='extr', params=extr_bests)

        # Change dictionary structure
        fitted = self._chg_dict_structure(fitted=fitted)

        for pred_day in pred_days:
            self._pred(pred_day=pred_day, apply_day=apply_day, fitted_model=fitted)

        print("Model 2 Prediction is finished")

    #################################
    # Methods for Data Preprocessing
    #################################
    # Load reservation history
    def _load_data_hx(self):
        res_hx = pd.read_csv(os.path.join(self.path_input, 'reservation', 'res_hx.csv'),
                             dtype={'res_num': str, 'res_route_nm': str, 'res_model_nm': str, 'rent_day': str,
                                    'rent_time': str, 'return_day': str, 'return_time': str, 'car_rent_fee': int,
                                    'cdw_fee': int, 'tot_fee': int, 'discount': float, 'res_day': str,
                                    'seasonality': int})

        return res_hx

    @staticmethod
    def _cluster_by_group(df: pd.DataFrame):
        av_ad = ['아반떼 AD (G)', '아반떼 AD (G) F/L']
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

        return df

    @staticmethod
    def _calc_exp_res_period(df: pd.DataFrame):
        # Change data types
        df['rent_datetime'] = df['rent_day'] + ' ' + df['rent_time']
        df['return_datetime'] = df['return_day'] + ' ' + df['return_time']
        df['rent_datetime'] = pd.to_datetime(df['rent_datetime'], format='%Y-%m-%d %H:%M:%S')
        df['return_datetime'] = pd.to_datetime(df['return_datetime'], format='%Y-%m-%d %H:%M:%S')
        df['discount'] = df['discount'].astype(int)

        # Caculate rent periods
        df['rent_period'] = df['return_datetime'] - df['rent_datetime']
        df['rent_period_hours'] = df['rent_period'].to_numpy().astype('timedelta64[h]') / np.timedelta64(1, 'h')

        return df

    def _calc_exp_sales(self, df: pd.DataFrame):
        df['fee_group'] = df['res_model'].apply(lambda x: self.fee_group[x])
        df['rent_day'] = df['rent_period_hours'] // 24
        df['rent_hour'] = df['rent_period_hours'] % 24

        df['rent_day'] = df['rent_day'].astype(int)
        df['rent_hour'] = df['rent_hour'].astype(int)

        df['rent_fee'] = df[['fee_group', 'rent_day', 'rent_hour']].apply(self._calc_fee, kinds='fee', axis=1)
        df['rent_cdw'] = df[['fee_group', 'rent_day', 'rent_hour']].apply(self._calc_fee, kinds='cdw', axis=1)

        return df

    def _calc_fee(self, x, kinds):
        fee_group = self.fee_per_times[kinds][x[0]]
        days = x[1]
        hours = x[2]
        if kinds == 'fee':
            return fee_group[24] * days + fee_group[hours]
        elif kinds == 'cdw':
            return fee_group[days]

    ###########################################
    # Common Methods for Training & Prediction
    ###########################################
    def _load_hx(self):
        # Load avg. reservation periods dataset
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

        # Load history trend
        res_data_hx = defaultdict(dict)
        data_types = copy.deepcopy(self.data_type)
        data_types.remove('disc')
        for data_type in data_types:
            for model in self.model_type:
                res_data_hx[data_type].update({model: pd.read_csv(os.path.join(self.path_trend_hx, data_type,
                                                                               data_type + '_' + model + '.csv'))})
        # Reservation discount
        res_data_hx['disc'] = res_data_hx['cnt_cum']

        self.res_data_hx = res_data_hx

    def _set_split_map(self):
        split_map = {'cnt_inc': {'drop': 'cnt_add',
                                 'target': 'cnt_add'},
                     'cnt_cum': {'drop': 'cnt_cum',
                                 'target': 'cnt_cum'},
                     'util_inc': {'drop': ['util_add', 'util_rate_add'],
                                  'target': 'util_rate_add'},
                     'util_cum': {'drop': ['util_cum', 'util_rate_cum'],
                                  'target': 'util_rate_cum'},
                     'disc': {'drop': ['disc_mean'],
                              'target': 'disc_mean'}}

        self.split_map = split_map

    def _split_input_target_all(self):
        io = {}
        for data_type in self.data_type:
            io_model = {}
            for model in self.model_type:
                split = self._split_input_target(data_type=data_type, model=model)
                io_model[model] = split
            io[data_type] = io_model

        return io

    def _split_input_target(self, data_type: str, model: str):
        x = self.res_data_hx[data_type][model].drop(columns=self.split_map[data_type]['drop'])
        y = self.res_data_hx[data_type][model][self.split_map[data_type]['target']]

        return {'x': x, 'y': y}

    def _split_train_test_all(self, data: dict):
        result = {}
        for type_key, type_val in data.items():    # Data type: cnt / util
            model_dict = {}
            for model_key, model_val in type_val.items():    # Model type: av_ad/ av_new / k3 / soul / vlst
                train, test = self._split_train_test(x=model_val['x'], y=model_val['y'])
                model_dict[model_key] = {'train': train, 'test': test}
            result[type_key] = model_dict

        return result

    def _split_train_test(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_state, shuffle=True)
        train = {'x': x_train, 'y': y_train}
        test = {'x': x_test, 'y': y_test}

        return train, test

    ##################################
    # Methods for Training
    ##################################
    def _grid_search_cross_validation(self, data: dict, regr: str):
        # Grid search cross validation
        regr_bests = {}
        for type_key, type_val in data.items():    # type_key: cnt / disc / util
            model_bests = {}
            for model_key, model_val in type_val.items():    # model_key: av / k3 / vl
                train = model_val['train']
                regr_best = self._get_best_hyper_param(x=train['x'], y=train['y'],
                                                       regr=regr,
                                                       regressors=self.REGRESSORS,
                                                       param_grids=self.param_grids)
                print(f'Data: {type_key} / Model: {model_key} is trained')
                model_bests[model_key] = regr_best
            regr_bests[type_key] = model_bests

        return regr_bests

    @staticmethod
    def _get_best_hyper_param(x, y, regressors, param_grids, regr, scoring='neg_root_mean_squared_error'):
        # Select regressor algorithm
        regressor = regressors[regr]

        # Define parameter grid
        param_grid = param_grids[regr]

        # Initialize Grid Search object
        gscv = GridSearchCV(estimator=regressor, param_grid=param_grid,
                            scoring=scoring, n_jobs=1, cv=5, verbose=1)

        # Fit gscv
        print(f'Tuning {regr}')
        gscv.fit(x, y)

        # Get best paramters and score
        best_params = gscv.best_params_
        # best_score = gscv.best_score_

        # Update regressor paramters
        regressor.set_params(**best_params)

        return regressor

    def _save_best_params(self, regr: str, regr_bests: dict):
        for type_key, type_val in regr_bests.items():
            for model_key, model_val in type_val.items():
                best_params = model_val.get_params()
                f = open(os.path.join(self.path_model, type_key, regr + '_params_' + model_key + '.pickle'), 'wb')
                pickle.dump(best_params, f)
                f.close()

    ##################################
    # Methods for Prediction
    ##################################
    def _set_pred_init_variables(self):
        self.day_to_season = self._get_seasonal_map()
        self.disc_re = self._get_init_disc_map()
        self.capa_re = self._get_init_capa()
        self.lt, self.lt_vec, self.lt_to_lt_vec = self._get_lead_time()

    @staticmethod
    def _get_seasonal_map():
        # Load recent seasonality dataset
        load_path = os.path.join('..', 'input', 'seasonality')
        ss_curr = pd.read_csv(os.path.join(load_path, 'seasonality_curr.csv'), delimiter='\t')
        ss_curr['date'] = pd.to_datetime(ss_curr['date'], format='%Y%m%d')
        day_to_season = {day: season for day, season in zip(ss_curr['date'], ss_curr['seasonality'])}

        return day_to_season

    @staticmethod
    def _get_init_disc_map():
        # Initial discount rate for each season
        load_path = os.path.join('..', 'input', 'discount')
        dscnt_init = pd.read_csv(os.path.join(load_path, 'discount_init.csv'), delimiter='\t')
        dscnt_init['date'] = pd.to_datetime(dscnt_init['date'], format='%Y%m%d')
        day_to_init_discount = {day: discount for day, discount in zip(dscnt_init['date'],
                                                                       dscnt_init['discount_init'])}

        return day_to_init_discount

    @staticmethod
    def _get_init_capa():
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'capa')
        capa_init = pd.read_csv(os.path.join(load_path, 'capa_curr_car.csv'), delimiter='\t',
                                dtype={'date': str, 'model': str, 'capa': int})
        mon_to_capa_init = {(date, model): capa for date, model, capa in zip(capa_init['date'],
                                                                             capa_init['model'],
                                                                             capa_init['capa'])}

        return mon_to_capa_init

    @staticmethod
    def _get_lead_time():
        # Lead Time Setting
        lt = np.arange(-89, 1, 1)
        lt_vec = np.arange(-36, 1, 1)
        lt_to_lt_vec = {-1 * i: (((i // 7) + 24) * -1 if i > 28 else i * -1) for i in range(0, 90, 1)}

        return lt, lt_vec, lt_to_lt_vec

    def _get_init_res_cnt(self):
        # Load recent reservation dataset
        load_path = os.path.join('..', 'input', 'reservation')
        res_re = pd.read_csv(os.path.join(load_path, 'res_' + self.res_update_day + '.csv'), delimiter='\t')

        # Rename columns
        res_remap_cols = {
            '예약경로': 'res_route', '예약경로명': 'res_route_nm', '계약번호': 'res_num',
            '고객구분': 'cust_kind', '고객구분명': 'cust_kind_nm', '총 청구액(VAT포함)': 'tot_fee',
            '예약모델': 'res_model', '예약모델명': 'res_model_nm', '차급': 'car_grd',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '대여기간(일)': 'rent_period_day', '대여기간(시간)': 'rent_period_time',
            'CDW요금': 'cdw_fee', '할인유형': 'discount_type', '할인유형명': 'discount_type_nm',
            '적용할인명': 'applyed_discount', '적용할인율(%)': 'discount', '회원등급': 'member_grd',
            '구매목적': 'sale_purpose', '생성일': 'res_day', '차종': 'car_kind'}
        res_re = res_re.rename(columns=res_remap_cols)

        # filter only 1.6 grade car group
        res_re = res_re[res_re['res_model_nm'].isin([
            '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)', 'ALL NEW K3 (G)',
            '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)'])]

        # Group Car Model
        conditions = [
            res_re['res_model_nm'] == '아반떼 AD (G) F/L',
            res_re['res_model_nm'] == '올 뉴 아반떼 (G)',
            res_re['res_model_nm'] == 'ALL NEW K3 (G)',
            res_re['res_model_nm'].isin(['쏘울 (G)', '쏘울 부스터 (G)']),
            res_re['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
        values = self.model_type
        res_re['res_model_grp'] = np.select(conditions, values)

        # Drop columns 1
        res_re = res_re.drop(columns=['res_model_nm'], errors='ignore')
        res_re = res_re.sort_values(by=['rent_day', 'res_day'])

        res_util = self._get_res_util(df=res_re)

        # Drop columns 2
        res_drop_col = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm',
                        'res_model', 'car_grd', 'rent_time', 'return_day', 'return_time', 'rent_period_day',
                        'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm', 'sale_purpose',
                        'applyed_discount', 'discount_rate', 'member_grd', 'sale_purpose', 'car_kind']
        res_re = res_re.drop(columns=res_drop_col, errors='ignore')

        res_re['rent_day'] = pd.to_datetime(res_re['rent_day'], format='%Y-%m-%d')
        res_re['res_day'] = pd.to_datetime(res_re['res_day'], format='%Y-%m-%d')

        # Filter datetime
        res_re = res_re[res_re['rent_day'] <= dt.datetime(2021, 2, 28)]
        res_util = res_util[res_util['rent_day'] <= dt.datetime(2021, 2, 28)]

        return self._group(res_cnt=res_re, res_util=res_util)

    def _group(self, res_cnt: pd.DataFrame, res_util: pd.DataFrame):
        # Grouping
        cum_cnt = res_cnt.groupby(by=['rent_day', 'res_model_grp']).count()['res_num']
        cum_cnt = cum_cnt.reset_index(level=(0, 1))

        res_cnt_dict = defaultdict(dict)
        for day, model, cnt in zip(cum_cnt['rent_day'], cum_cnt['res_model_grp'], cum_cnt['res_num']):
            day_str = day.strftime('%Y-%m-%d')
            res_cnt_dict[day_str].update({model: cnt})

        cum_util = res_util.groupby(by=['rent_day', 'res_model_grp']).sum()['util_rate']
        cum_util = cum_util.reset_index(level=(0, 1))

        res_util_dict = defaultdict(dict)
        for day, model, util in zip(cum_util['rent_day'], cum_util['res_model_grp'], cum_util['util_rate']):
            mon_str = day.strftime('%Y%m')
            day_str = day.strftime('%Y-%m-%d')
            res_util_dict[day_str].update({model: util / self.capa_re[(mon_str, self.model_type_map[model])]})

        cum_sales = res_cnt.groupby(by=['rent_day', 'res_model_grp']).sum()['tot_fee']
        cum_sales = cum_sales.reset_index(level=(0, 1))

        res_sales_dict = defaultdict(dict)
        for day, model, sales in zip(cum_sales['rent_day'], cum_sales['res_model_grp'], cum_sales['tot_fee']):
            day_str = day.strftime('%Y-%m-%d')
            res_sales_dict[day_str].update({model: sales})

        return res_cnt_dict, res_util_dict, res_sales_dict

    @staticmethod
    def _get_res_util(df: pd.DataFrame):
        res_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                df['rent_day'], df['rent_time'], df['return_day'], df['return_time'],
                df['res_day'], df['discount'], df['res_model_grp']):

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
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util_rate', 'discount', 'res_model_grp'])

        return res_util_df

    def _load_best_params(self, regr: str):
        regr_bests = {}
        for data_type in self.data_type:
            model_bests = {}
            for model in self.model_type:
                f = open(os.path.join(self.path_model, data_type, regr + '_params_' + model + '.pickle'), 'rb')
                model_bests[model] = pickle.load(f)
                f.close()
            regr_bests[data_type] = model_bests

        return regr_bests

    def _fit_model(self, dataset: dict, regr: str, params: dict):
        fitted = {}
        for type_key, type_val in params.items():
            model_fit = {}
            for model_key, model_val in type_val.items():
                model = self.REGRESSORS[regr](**model_val)
                data = dataset[type_key][model_key]
                model.fit(data['x'], data['y'])
                model_fit[model_key] = model
            fitted[type_key] = model_fit

        return fitted

    @staticmethod
    def _chg_dict_structure(fitted: dict):
        fitted_re = defaultdict(dict)
        for type_key, type_val in fitted.items():
            for model_key, model_val in type_val.items():
                fitted_re[model_key].update({type_key: model_val})

        return fitted_re

    def _get_pred_input_init(self, pred_mon: str, season: int, lead_time_vec: int, res_cnt: dict, disc: int):
        pred_input = defaultdict(dict)
        for model in self.model_type:
            capa = self.capa_re[(pred_mon, self.model_type_map[model])]
            for data_type in self.data_type:
                if data_type in ['disc']:
                    pred_input[model].update({data_type: np.array([season, lead_time_vec,
                                                                   res_cnt.get(model, 0) / capa]).reshape(1, -1)})
                else:
                    pred_input[model].update({data_type: np.array([season, lead_time_vec, disc]).reshape(1, -1)})

        return pred_input

    def _get_pred_input(self, season: int, lead_time_vec: int, res_cnt: int, disc: int, capa: int):
        pred_input = defaultdict(dict)
        for data_type in self.data_type:
            if data_type in ['disc']:
                pred_input[data_type] = np.array([season, lead_time_vec, res_cnt / capa]).reshape(1, -1)
            else:
                pred_input[data_type] = np.array([season, lead_time_vec, disc]).reshape(1, -1)

        return pred_input

    def _pred_sales_exp(self, pred_mon: str, pred_input: dict, season: int, init_data: list, lead_time_vec: int,
                        fitted_model: dict, sales_per_res: dict, cdw_per_res: dict):
        # Calculate first row
        temp = defaultdict(dict)
        for model, model_val in fitted_model.items():   # Model: ad_av / ad_new / k3 / soul / vlst
            # Set current reservation status
            res_cnt = init_data[0].get(model, 0)    # Current reservation counts
            disc = init_data[2]                     # Current discount
            exp_sales = init_data[3].get(model, 0)  # Current sales
            temp[model].update({'lead_time_vec': [lead_time_vec]})
            temp[model].update({'res_cnt': [res_cnt]})
            temp[model].update({'res_util': [round(init_data[1].get(model, 0), 1)]})
            temp[model].update({'disc': [disc]})
            temp[model].update({'exp_sales': [exp_sales]})    # Expected sales
            temp[model].update({'exp_cum_sales': [exp_sales]})    # Expected cum. sales

            capa = self.capa_re[(pred_mon, self.model_type_map[model])]

            # Calculate current sales
            for key, val in model_val.items():    # Type: cnt_inc / cnt_cum / util_inc / util_cum / disc
                if key in ['cnt_inc', 'cnt_cum']:
                    temp[model].update({'exp_' + key: [round(val.predict(pred_input[model][key])[0] * capa, 2)]})
                else:
                    temp[model].update({'exp_' + key: [round(val.predict(pred_input[model][key])[0], 2)]})

        # Calculate continuous rows
        for lt_vec in np.arange(lead_time_vec + 1, 1):
            for model, model_val in fitted_model.items():
                temp[model]['lead_time_vec'].append(lt_vec)

                # Calculate next reservation status
                cnt = temp[model]['res_cnt'][-1] + temp[model]['exp_cnt_inc'][-1]
                util = temp[model]['res_util'][-1] + temp[model]['exp_util_inc'][-1]
                disc = temp[model]['exp_disc'][-1]
                exp_sales = temp[model]['exp_cnt_inc'][-1] * (sales_per_res[model] * (1-disc/100) + cdw_per_res[model])

                # Maximum Utilization rate : 100%
                if util >= 1:
                    util = 1
                    if temp[model]['res_util'][-1] >= 1:
                        exp_sales = 0
                exp_cum_sales = temp[model]['exp_cum_sales'][-1] + exp_sales

                # Rounding
                temp[model]['res_cnt'].append(round(cnt, 1))
                temp[model]['res_util'].append(round(util, 2))
                temp[model]['disc'].append(round(disc, 1))
                temp[model]['exp_sales'].append(round(exp_sales, 0))
                temp[model]['exp_cum_sales'].append(round(exp_cum_sales, 0))

                capa = self.capa_re[(pred_mon, self.model_type_map[model])]

                # Expectation
                pred_temp = self._get_pred_input(season=season, lead_time_vec=lt_vec, res_cnt=cnt, disc=disc, capa=capa)
                for type_key, type_val in model_val.items():
                    if type_key in ['cnt_inc', 'cnt_cum']:
                        temp[model]['exp_' + type_key].append(round(type_val.predict(pred_temp[type_key])[0] * capa, 2))
                    else:
                        temp[model]['exp_' + type_key].append(round(type_val.predict(pred_temp[type_key])[0], 2))

        return temp

    def _pred_sales_rec(self, pred_mon: str, pred_input: dict, season: int, init_data: list, lead_time_vec: int,
                        fitted_model: dict, sales_per_res: dict, cdw_per_res: dict):
        # Calculate first row
        temp = defaultdict(dict)
        for model, model_val in fitted_model.items():   # Model: ad_av / ad_new / k3 / soul / vlst
            # Set current reservation status
            res_cnt = init_data[0].get(model, 0)    # Current reservation counts
            res_util = round(init_data[1].get(model, 0), 1)
            disc = init_data[2]                     # Current discount
            exp_sales = init_data[3].get(model, 0)  # Current sales
            temp[model].update({'lead_time_vec': [lead_time_vec]})
            temp[model].update({'res_cnt': [res_cnt]})
            temp[model].update({'res_util': [res_util]})
            temp[model].update({'disc': [disc]})
            temp[model].update({'exp_sales': [exp_sales]})    # Expected sales
            temp[model].update({'exp_cum_sales': [exp_sales]})    # Expected cum. sales

            capa = self.capa_re[(pred_mon, self.model_type_map[model])]
            # Calculate current sales
            for key, val in model_val.items():    # Type: cnt_inc / cnt_cum / util_inc / util_cum / disc
                if key in ['cnt_inc', 'cnt_cum']:
                    temp[model].update({'exp_' + key: [round(val.predict(pred_input[model][key])[0] * capa, 2)]})
                else:
                    temp[model].update({'exp_' + key: [round(val.predict(pred_input[model][key])[0], 2)]})

            # Calculate recommendation discount
            disc_chg = self._rec_disc_function(curr=res_util * 100, exp=temp[model]['exp_util_cum'][-1] * 100)
            disc_rec = disc * (1 + disc_chg / 100)

            # Apply discount policy
            disc_rec = self._apply_disc_policy(disc=disc_rec)
            temp[model].update({'rec_disc': [round(disc_rec, 1)]})

        # Calculate continuous rows
        for lt_vec in np.arange(lead_time_vec + 1, 1):
            for model, model_val in fitted_model.items():
                temp[model]['lead_time_vec'].append(lt_vec)

                # Calculate next reservation status
                cnt = temp[model]['res_cnt'][-1] + temp[model]['exp_cnt_inc'][-1]
                util = temp[model]['res_util'][-1] + temp[model]['exp_util_inc'][-1]
                disc = temp[model]['rec_disc'][-1]
                exp_sales = temp[model]['exp_cnt_inc'][-1] * (sales_per_res[model] * (1-disc/100) + cdw_per_res[model])

                # Maximum Utilization rate : 100%
                if util >= 1:
                    util = 1
                    if temp[model]['res_util'][-1] >= 1:
                        exp_sales = 0

                exp_cum_sales = temp[model]['exp_cum_sales'][-1] + exp_sales

                # Rounding
                temp[model]['res_cnt'].append(round(cnt, 1))
                temp[model]['res_util'].append(round(util, 2))
                temp[model]['disc'].append(round(disc, 1))
                temp[model]['exp_sales'].append(round(exp_sales, 0))
                temp[model]['exp_cum_sales'].append(round(exp_cum_sales, 0))

                capa = self.capa_re[(pred_mon, self.model_type_map[model])]

                # Expectation
                pred_temp = self._get_pred_input(season=season, lead_time_vec=lt_vec, res_cnt=cnt, disc=disc, capa=capa)
                for type_key, type_val in model_val.items():
                    if type_key in ['cnt_inc', 'cnt_cum']:
                        temp[model]['exp_' + type_key].append(round(type_val.predict(pred_temp[type_key])[0] * capa, 2))
                    else:
                        temp[model]['exp_' + type_key].append(round(type_val.predict(pred_temp[type_key])[0], 2))

                # Calculate recommendation discount
                disc_chg = self._rec_disc_function(curr=util * 100, exp=temp[model]['exp_util_cum'][-1] * 100)
                disc_rec = disc * (1 + disc_chg / 100)

                # Apply discount policy
                disc_rec = self._apply_disc_policy(disc=disc_rec)
                temp[model]['rec_disc'].append(round(disc_rec, 1))

        return temp

    @staticmethod
    def _apply_disc_policy(disc: int):
        # Set maximum discount rate: 80%
        if disc > 80:
            disc = 80
        # Convert discount to five times values
        if disc % 5 >= 2.5:
            return (disc // 5 + 1) * 5
        else:
            return (disc // 5) * 5

    def _save_result(self, result: dict, pred_day: str, apply_day: str, kinds: str):
        pred_day_str = ''.join(pred_day.split('-'))
        apply_day_str = ''.join(apply_day.split('/'))

        save_path = os.path.join(self.path_sales_per_res, apply_day_str)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(os.path.join(save_path, kinds)):
            os.mkdir(os.path.join(save_path, kinds))

        for model_key, model_val in result.items():
            save_path_day = os.path.join(save_path, kinds, pred_day_str)
            if not os.path.exists(save_path_day):
                os.mkdir(save_path_day)
            pd.DataFrame(model_val).T.to_csv(os.path.join(save_path_day, 'sales_pred_' + model_key + '.csv'),
                                             header=False)

    @staticmethod
    def _rec_disc_function(curr: float, exp: float, dmd=0):
        """
        Customized Exponential function
        curr_util: Current Utilization Rate
        d: Exp Demand Change Rate
        ex_util: Exp. Utilization rate
        """

        # Hyperparamters (need to tune)
        theta1 = 1          # Ratio of Decreasing magnitude
        if curr < exp:      # Ratio of increasing
            theta1 = 0.5
        theta2 = 0.05  # ratio of increasing / decreasing magnitude : demand
        phi_low = 1.7  # 1 < phi_low < phi_high < 2
        phi_high = 1.2  # 1 < phi_low < phi_high < 2

        if dmd > 0:
            y = 1 - theta1 * (curr ** (phi_high ** (-1 * (curr * theta2 * dmd))) - exp)
        else:
            y = 1 - theta1 * (curr ** (phi_low ** (-1 * ((1 - curr) * theta2 * dmd))) - exp)

        return y

    def _get_rec(self, x):
        return self._rec_disc_function(curr=x[0], exp=x[1], dmd=0)

    def _pred(self, pred_day: str, apply_day: str, fitted_model: dict):
        # Get season value and initial discount rate
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        pred_mon = ''.join(pred_day.split('-'))[:6]
        apply_datetime = dt.datetime(*list(map(int, apply_day.split('/'))))
        season = self.day_to_season[pred_datetime]
        fee_per_res = self.rent_fee_hx[season]  # Reservation fee of history dataset by each model
        cdw_per_res = self.rent_cdw_hx[season]
        init_disc = self.disc_re[pred_datetime]
        init_res = self.res_cnt_init[pred_day]
        init_util = self.res_util_init[pred_day]
        init_sales = self.res_sales_init[pred_day]

        # Calculate lead time vector
        lead_time = (pred_datetime - apply_datetime).days
        lead_time_vec = self.lt_to_lt_vec[lead_time * -1]

        # Make initial input
        pred_input = self._get_pred_input_init(pred_mon=pred_mon, season=season, lead_time_vec=lead_time_vec,
                                               res_cnt=init_res, disc=init_disc)

        # Rolling sales prediction
        # Expect sales on past trend
        pred_sales_exp = self._pred_sales_exp(pred_mon=pred_mon, pred_input=pred_input, season=season,
                                              lead_time_vec=lead_time_vec,
                                              sales_per_res=fee_per_res, cdw_per_res=cdw_per_res,
                                              init_data=[init_res, init_util, init_disc, init_sales],
                                              fitted_model=fitted_model)
        # Expect sales on recommendation
        pred_sales_rec = self._pred_sales_rec(pred_mon=pred_mon, pred_input=pred_input, season=season,
                                              lead_time_vec=lead_time_vec,
                                              sales_per_res=fee_per_res, cdw_per_res=cdw_per_res,
                                              init_data=[init_res, init_util, init_disc, init_sales],
                                              fitted_model=fitted_model)

        # Rearrange and drop unncessary columns
        pred_sales_exp = self._rearr_column(df=pred_sales_exp, pred_day=pred_day, apply_day=apply_day, kinds='exp')
        pred_sales_rec = self._rearr_column(df=pred_sales_rec, pred_day=pred_day, apply_day=apply_day, kinds='rec')

        self._save_result(result=pred_sales_exp, pred_day=pred_day, apply_day=apply_day, kinds='exp')
        self._save_result(result=pred_sales_rec, pred_day=pred_day, apply_day=apply_day, kinds='rec')

        # self._draw_trend_graph(exp=pred_sales_exp, rec=pred_sales_rec,
        #                        pred_day=pred_day, apply_day=apply_day)

        print(f'Prediction result on {pred_day} is saved')

    def _draw_trend_graph(self, exp: dict, rec: dict, pred_day: str, apply_day: str):
        pred_day_str = ''.join(pred_day.split('-'))
        apply_day_str = ''.join(apply_day.split('/'))

        fig_size = (9, 6)
        for (exp_key, exp_val), (rec_key, rec_val) in zip(exp.items(), rec.items()):
            # Figure setting
            fig, axes = plt.subplots(3, 1)
            cor_exp = '#3a18b1'
            cor_exp_exp = '#045c5a'
            cor_rec = '#9d0216'
            cor_rec_exp = '#fb5581'

            # Dataframe setting
            exp_sales = exp_val['exp_cum_sales'].iloc[-1]
            rec_sales = rec_val['exp_cum_sales'].iloc[-1]

            exp_val = exp_val.set_index('lead_time', drop=True)
            rec_val = rec_val.set_index('lead_time', drop=True)

            # Reservation Count Graph
            exp_val['cnt'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                     color=cor_exp, label='예약건수(exp)', ax=axes[0])
            exp_val['exp_cnt'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9, linestyle='--',
                                         color=cor_exp_exp, label='기대 예약건수(exp)', ax=axes[0])
            rec_val['cnt'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                     color=cor_rec, label='예약건수(rec)', ax=axes[0])
            rec_val['exp_cnt'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9, linestyle='--',
                                         color=cor_rec_exp, label='기대 예약건수(rec)', ax=axes[0])
            axes[0].set_xlabel('리드타임')
            axes[1].set_ylabel('건')
            axes[0].legend()

            # Reservation Utilization Graph
            exp_val['util'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                      color=cor_exp, label='가동률(exp)', ax=axes[1])
            exp_val['exp_util'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                          color=cor_exp_exp, label='기대 가동률(exp)', ax=axes[1])
            rec_val['util'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                      color=cor_rec, label='가동률(rec)', ax=axes[1])
            rec_val['exp_util'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                          color=cor_rec_exp, label='기대 가동률(rec)', ax=axes[1])
            axes[1].set_xlabel('리드타임')
            axes[1].set_ylabel('%')
            axes[1].legend()

            # Reservation Discount Graph
            exp_val['disc'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                      color=cor_exp, label='할인율(exp)', ax=axes[2])
            exp_val['exp_disc'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                          color=cor_exp_exp, label='기대 할인율(exp)', ax=axes[2])
            rec_val['disc'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                      color=cor_rec, label='할인율(rec)', ax=axes[2])
            rec_val['rec_disc'].plot.line(figsize=fig_size, linewidth=0.9, alpha=0.9,
                                          color=cor_rec_exp, label='추천 할인율(rec)', ax=axes[2])
            axes[2].set_xlabel('리드타임')
            axes[2].set_ylabel('%')
            axes[2].legend()

            plt.suptitle(f'Exp. Sales: {exp_sales} / Rec. Sales: {rec_sales}')

            # Save images
            save_path = os.path.join(self.path_sales_per_res, apply_day_str, 'img', pred_day_str)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(os.path.join(save_path, 'sales_pred_' + exp_key + '.png'))
            plt.close()

    @staticmethod
    def _rearr_column(df: dict, pred_day: str, apply_day: str, kinds: str):
        start_day = dt.datetime.strptime(apply_day, "%Y/%m/%d")
        end_day = dt.datetime.strptime(pred_day, "%Y-%m-%d")

        if (end_day - start_day).days < 28:
            days = pd.date_range(start=start_day, end=end_day, freq='D')
        else:
            lt_four_week = end_day - relativedelta(days=26)
            days_after = pd.date_range(start=lt_four_week, end=end_day, freq='D')
            periods = (((end_day - start_day).days - 28) // 7) + 1
            days = [start_day]
            days.extend(pd.date_range(end=lt_four_week - relativedelta(days=1), periods=periods, freq='7D'))
            days.extend(days_after)

        days = pd.Series(days).dt.strftime('%Y-%m-%d')

        arr_cols = []
        if kinds == 'exp':
            arr_cols = ['day', 'lead_time_vec', 'res_cnt', 'res_util', 'disc',
                        'exp_cnt_cum', 'exp_util_cum', 'exp_disc', 'exp_sales', 'exp_cum_sales']
        elif kinds == 'rec':
            arr_cols = ['day', 'lead_time_vec', 'res_cnt', 'res_util', 'disc',
                        'exp_cnt_cum', 'exp_util_cum', 'exp_disc', 'rec_disc', 'exp_sales', 'exp_cum_sales']

        rename_cols = {'lead_time_vec': 'lead_time', 'res_cnt': 'cnt', 'res_util': 'util',
                       'exp_cnt_cum': 'exp_cnt', 'exp_util_cum': 'exp_util'}

        results = {}
        for model_key, model_val in df.items():
            model_df = pd.DataFrame(model_val)
            model_df['day'] = days
            model_df = model_df[arr_cols]
            model_df = model_df.rename(columns=rename_cols)
            results[model_key] = model_df

        return results
