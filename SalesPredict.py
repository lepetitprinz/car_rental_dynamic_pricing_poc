import os
import copy
import pickle
import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor


class SalesPredict(object):

    REGRESSORS = {"Extra Trees Regressor": ExtraTreesRegressor(),
                  "extr": ExtraTreesRegressor}

    def __init__(self, res_update_day: str):
        self.random_state = 2020
        self.test_size = 0.2
        self.data_type: list = ['cnt_inc', 'cnt_cum', 'util_inc', 'util_cum', 'disc']
        self.model_type: list = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']
        self.model_type_map: dict = {'av_ad': '아반떼 AD (G) F/L', 'av_new': '올 뉴 아반떼 (G)',
                                     'k3': 'ALL NEW K3 (G)', 'soul': '쏘울 부스터 (G)',
                                     'vlst': '더 올 뉴 벨로스터 (G)'}

        # Path of data & model
        self.path_data = os.path.join('..', 'result', 'data', 'model_2', 'hx', 'car')
        self.path_model = os.path.join('..', 'result', 'model', 'res_pred_lead_time')

        self.res_data_hx: dict = {}

        # inintialize mapping dictionary
        self.data_map: dict = dict()
        self.split_map: dict = dict()
        self.param_grids = dict()

        # Prediction variables
        # Initial values of variables
        self.res_update_day = res_update_day
        self.day_to_init_res_cnt: dict = {}
        self.day_to_init_res_util: dict = {}
        self.day_to_season: dict = {}
        self.day_to_disc_init: dict = {}
        self.mon_to_capa_init: dict = {}
        self.avg_unavail_capa = 2
        # Lead time
        self.lt: np.array = []
        self.lt_vec: np.array = []
        self.lt_to_lt_vec: dict = {}

    def train(self):
        # Load dataset
        res_hx = self._load_data_hx()

        # Define input and output
        self._set_split_map(data=res_hx)

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
        # Set initial variables
        self._set_pred_init_variables()

        # Set recent data mapping
        self.day_to_init_res_cnt, self.day_to_init_res_util = self._get_init_res_cnt()

        # Load history dataset
        res_hx = self._load_data_hx()

        # Define input and output
        self._set_split_map(data=res_hx)

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

    def _pred(self, pred_day: str, apply_day: str, fitted_model: dict):
        # Get season value and initial discount rate
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        pred_mon = pred_day.split('-')[0] + pred_day.split('-')[1]
        apply_datetime = dt.datetime(*list(map(int, apply_day.split('/'))))
        season = self.day_to_season[pred_datetime]
        init_disc = self.day_to_disc_init[pred_datetime]
        init_res = self.day_to_init_res_cnt[pred_day]
        init_util = self.day_to_init_res_util[pred_day]
        init_capa = {key: self.mon_to_capa_init[(pred_mon, val)] - self.avg_unavail_capa for key,
                                                                                    val in self.model_type_map.items()}
        # Calculate lead time vector
        lead_time = (pred_datetime - apply_datetime).days
        lead_time_vec = self.lt_to_lt_vec[lead_time * -1]

        # Make initial values dataframe
        pred_input = self._get_pred_input_init(season=season, lead_time_vec=lead_time_vec,
                                               res_cnt=init_res, disc=init_disc)
        pred_result = self._pred_fitted_model(pred_input=pred_input,
                                              season=season,
                                              init_data=[init_res, init_util, init_disc],
                                              lead_time_vec=lead_time_vec,
                                              fitted_model=fitted_model)

        # Map lead time to prediction results
        lt_to_pred_result = self._get_lt_to_pred_result(pred_result=pred_result)
        result = self._map_rslt_to_lead_time(pred_final=lt_to_pred_result)

        # Result data convert to dataframe
        result_df = self._conv_to_dataframe(result=result, pred_datetime=pred_datetime,
                                            init_disc=init_disc, init_capa=init_capa)

        # Save the result dataframe
        self._save_result(result=result_df, pred_day=pred_day)

        print(f'Prediction result on {pred_day} is saved')

    ####################################
    # 2. Data & Variable Initialization
    ####################################
    def _load_data_hx(self):
        res_data_hx = defaultdict(dict)
        data_types = copy.deepcopy(self.data_type)
        data_types.remove('disc')
        for data_type in data_types:
            for model in self.model_type:
                res_data_hx[data_type].update({model: pd.read_csv(os.path.join(self.path_data, data_type,
                                                               data_type + '_' + model + '.csv'))})
        # Reservation discunt
        res_data_hx['disc'] = res_data_hx['cnt_cum']

        self.res_data_hx = res_data_hx

        return res_data_hx

    def _set_split_map(self, data: dict):
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

    ##################################
    # 2. Data Preprcessing
    ##################################
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
    # 3. Training
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
    # 4. Prediction
    ##################################
    def _set_pred_init_variables(self):

        self.day_to_season = self._get_seasonal_map()
        self.day_to_disc_init = self._get_init_disc_map()
        self.mon_to_capa_init = self._get_init_capa()
        self.lt, self.lt_vec, self.lt_to_lt_vec = self._get_lead_time()

    def _get_init_res_cnt(self):
        # Load recent reservation dataset
        load_path = os.path.join('..', 'input', 'reservation')
        res_curr = pd.read_csv(os.path.join(load_path, 'res_' + self.res_update_day + '.csv'), delimiter='\t')

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
        res_curr = res_curr.rename(columns=res_remap_cols)

        # filter only 1.6 grade car group
        res_curr = res_curr[res_curr['res_model_nm'].isin([
            '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)', 'ALL NEW K3 (G)',
            '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)'])]

        # Group Car Model
        conditions = [
            res_curr['res_model_nm'] == '아반떼 AD (G) F/L',
            res_curr['res_model_nm'] == '올 뉴 아반떼 (G)',
            res_curr['res_model_nm'] == 'ALL NEW K3 (G)',
            res_curr['res_model_nm'].isin(['쏘울 (G)', '쏘울 부스터 (G)']),
            res_curr['res_model_nm'] == '더 올 뉴 벨로스터 (G)']
        values = self.model_type
        res_curr['res_model_grp'] = np.select(conditions, values)

        # Drop columns 1
        res_curr = res_curr.drop(columns=['res_model_nm'], errors='ignore')
        res_curr = res_curr.sort_values(by=['rent_day', 'res_day'])

        res_util = self._get_res_util(df=res_curr)

        # Drop columns 2
        res_drop_col = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm', 'tot_fee',
                        'res_model', 'car_grd', 'rent_time', 'return_day', 'return_time', 'rent_period_day',
                        'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm', 'sale_purpose',
                        'applyed_discount', 'discount_rate', 'member_grd', 'sale_purpose', 'car_kind']
        res_curr = res_curr.drop(columns=res_drop_col, errors='ignore')

        res_curr['rent_day'] = pd.to_datetime(res_curr['rent_day'], format='%Y-%m-%d')
        res_curr['res_day'] = pd.to_datetime(res_curr['res_day'], format='%Y-%m-%d')

        # Grouping
        cnt_cum = res_curr.groupby(by=['rent_day', 'res_model_grp']).count()['res_num']
        cnt_cum = cnt_cum.reset_index(level=(0, 1))

        res_cnt_dict = defaultdict(dict)
        for day, model, cnt in zip(cnt_cum['rent_day'], cnt_cum['res_model_grp'], cnt_cum['res_num']):
            day_str = day.strftime('%Y-%m-%d')
            res_cnt_dict[day_str].update({model: cnt})

        util_cum = res_util.groupby(by=['rent_day', 'res_model_grp']).sum()['util_rate']
        util_cum = util_cum.reset_index(level=(0, 1))

        res_util_dict = defaultdict(dict)
        for day, model, util in zip(util_cum['rent_day'], util_cum['res_model_grp'], util_cum['util_rate']):
            mon_str = day.strftime('%Y%m')
            day_str = day.strftime('%Y-%m-%d')
            res_util_dict[day_str].update({model: util / self.mon_to_capa_init[(mon_str, self.model_type_map[model])]})

        return res_cnt_dict, res_util_dict

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
            ft = timedelta(hours=fst[0], minutes=fst[1])  # time of rent day
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
                date_range,
                [res_day] * date_len,
                util,
                [discount] * date_len,
                [model] * date_len
            ]).T)
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util_rate', 'discount', 'res_model_grp'])

        return res_util_df

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
        lt = np.arange(-83, 1, 1)
        lt_vec = np.arange(-36, 1, 1)

        lt_to_lt_vec = {-1 * i: (((i // 7) + 24) * -1 if i > 28 else i * -1) for i in range(0, 84, 1)}

        return lt, lt_vec, lt_to_lt_vec

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

    def _get_pred_input_init(self, season: int, lead_time_vec: int, res_cnt: dict, disc: int):
        pred_input = defaultdict(dict)
        for model in self.model_type:
            for data_type in self.data_type:
                if data_type in ['disc']:
                    pred_input[model].update({data_type: np.array([season, lead_time_vec,
                                                                   res_cnt.get(model, 0)]).reshape(1, -1)})
                else:
                    pred_input[model].update({data_type: np.array([season, lead_time_vec, disc]).reshape(1, -1)})

        return pred_input

    def _get_pred_input(self, season: int, lead_time_vec: int, res_cnt: int, disc: int):
        pred_input = defaultdict(dict)
        for data_type in self.data_type:
            if data_type in ['disc']:
                pred_input[data_type] = np.array([season, lead_time_vec, res_cnt]).reshape(1, -1)
            else:
                pred_input[data_type] = np.array([season, lead_time_vec, disc]).reshape(1, -1)

        return pred_input

    def _pred_fitted_model(self, pred_input: dict, season: int, init_data: list, lead_time_vec: int,  fitted_model: dict):
        pred_results = {}

        # Calculate first row
        temp_dict = defaultdict(dict)
        for model_key, model_val in fitted_model.items():   # ad_av / ad_new / k3 / soul / vlst
            temp_dict[model_key].update({'lead_time_vec': [lead_time_vec]})
            temp_dict[model_key].update({'res_cnt': [init_data[0].get(model_key, 0)]})
            temp_dict[model_key].update({'res_util': [init_data[1].get(model_key, 0)]})
            temp_dict[model_key].update({'disc': [init_data[2]]})
            for type_key, type_val in model_val.items():    # cnt_inc / cnt_cum / util_inc / util_cum / disc
                temp_dict[model_key].update({'exp_' + type_key: [type_val.predict(pred_input[model_key][type_key])[0]]})

        # Calculate continuous rows
        for lt_vec in np.arange(lead_time_vec + 1, 1):
            for model_key, model_val in fitted_model.items():
                temp_dict[model_key]['lead_time_vec'].append(lt_vec)
                # Calculate
                cnt = temp_dict[model_key]['res_cnt'][-1] + temp_dict[model_key]['exp_cnt_inc'][-1]
                util = temp_dict[model_key]['res_util'][-1] + temp_dict[model_key]['exp_util_inc'][-1]
                disc = temp_dict[model_key]['exp_disc'][-1]
                temp_dict[model_key]['res_cnt'].append(round(cnt, 2))
                temp_dict[model_key]['res_util'].append(round(util, 2))
                temp_dict[model_key]['disc'].append(round(disc,2))

                # Expectation
                pred_temp = self._get_pred_input(season=season, lead_time_vec=lt_vec, res_cnt=cnt, disc=disc)
                for type_key, type_val in model_val.items():
                    temp_dict[model_key]['exp_' + type_key].append(type_val.predict(pred_temp[type_key])[0])

        return pred_results

    def _get_lt_to_pred_result(self, pred_result: dict):
        lt_to_pred_result = {}
        for type_key, type_val in pred_result.items():
            lt_to_models = {}
            for model_key, model_val in type_val.items():
                lt_to_pred = {lt: pred for lt, pred in zip(self.lt_vec, model_val)}
                lt_to_models[model_key] = lt_to_pred
            lt_to_pred_result[type_key] = lt_to_models

        return lt_to_pred_result

    def _map_rslt_to_lead_time(self, pred_final: dict):
        result = defaultdict(dict)
        for type_key, type_val in pred_final.items():   # type_key: cnt / disc / util
            for model_key, model_val in type_val.items():   # model_key: av / k3 / vl
                rslt = [model_val[self.lt_to_lt_vec[i]] for i in self.lt]
                result[model_key].update({type_key: rslt})

        return result

    def _conv_to_dataframe(self, result: dict, pred_datetime: dt.datetime,
                           init_disc: int, init_capa: dict):
        date = [pred_datetime - dt.timedelta(days=int(i*-1)) for i in self.lt]
        lead_time = self.lt
        curr_disc = init_disc

        model_df = {}
        for model_key, model_val in result.items():
            df = pd.DataFrame({'date': date, 'lead_time': lead_time, 'curr_disc': curr_disc})
            for type_key, type_val in model_val.items():
                df['exp_' + type_key] = type_val

            # Calculate utilization count
            df['exp_util_cnt'] = df['exp_util'] * init_capa[model_key]
            model_df[model_key] = df

        return model_df

    @staticmethod
    def _save_result(result: dict, pred_day: str):
        save_path = os.path.join('..', 'result', 'data', 'prediction')
        for model_key, model_val in result.items():
            model_val.to_csv(os.path.join(save_path, 'original', model_key,
                                          'm2_pred(' + pred_day + ').csv'), index=False)
