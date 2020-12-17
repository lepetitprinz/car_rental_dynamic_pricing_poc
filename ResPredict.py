from Utility import Utility

import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor


class ResPredict(object):

    REGRESSORS = {"Extra Trees Regressor": ExtraTreesRegressor(),
                  "extr": ExtraTreesRegressor}

    def __init__(self, res_status_ud_day: str, apply_day: str):
        self.utility = Utility
        self.apply_day = apply_day
        self.res_status_ud_day = res_status_ud_day

        # Path of data & model
        self.path_data_hx = os.path.join('..', 'result', 'data', 'model_2', 'hx')
        self.path_res_pred = os.path.join(self.utility.PATH_MODEL, 'res_prediction')

        # Data Type
        self.type_data = self.utility.TYPE_DATA
        self.type_data_map = self.utility.TYPE_DATA_MAP

        # Model type
        self.type_group = self.utility.TYPE_GROUP
        self.type_model = self.utility.TYPE_MODEL
        self.type_apply = {'model': self.type_group, 'car': self.type_model}
        self.model_nm_map = self.utility.MODEL_NAME_MAP
        self.avg_unavail_capa = self.utility.AVG_UNAVAIL_CAPA

        # Data preprocessing hyper-parameter
        self.test_size = self.utility.TEST_SIZE
        self.random_state = self.utility.RANDOM_STATE

        # Prediction variables
        # Initial values of variables
        self.res_rate_re: dict = {}
        self.season_re: dict = {}
        self.capacity_re: dict = {}
        self.disc_last_week: dict = {}

        # Lead time
        self.lt: np.array = []
        self.lt_vec: np.array = []
        self.lt_to_lt_vec: dict = {}

        # inintialize mapping dictionary
        self.data_map: dict = dict()
        self.param_grids = dict()
        self.split_map: dict = {'cnt': {'drop': ['cnt_cum'], 'target': 'cnt_cum'},
                                'disc': {'drop': ['disc_mean'], 'target': 'disc_mean'},
                                'util': {'drop': ['util_cum', 'util_rate_cum'], 'target': 'util_rate_cum'}}

    def train(self, type_apply: str):
        # Load dataset
        self.data_map = self._load_data(type_apply=type_apply)

        # Split into input and output
        applied_list = self.type_apply[type_apply]
        m2_io = self._split_input_target_all(applied_list=applied_list)

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
        self._save_best_params(regr='extr', regr_bests=extr_bests, type_apply=type_apply)

        print('Training finished')

    def predict(self, pred_days: list, disc_confirm_last_week: str, type_apply: str):
        applied_list = self.type_apply[type_apply]

        # Load dataset
        self.data_map = self._load_data(type_apply=type_apply)

        # Split into input and output
        m2_io = self._split_input_target_all(applied_list=applied_list)

        # Set initial variables
        self._set_recent_dataset(disc_confirm_last_week=disc_confirm_last_week, type_apply=type_apply)

        # Load best hyper-parameters
        extr_bests = self._load_best_params(regr='extr', type_apply=type_apply)

        # fit the model
        fitted = self._fit_model(dataset=m2_io, regressor='extr', params=extr_bests)

        for pred_day in pred_days:
            self._pred(pred_day=pred_day, fitted_model=fitted, applied_list=applied_list)

        print('')
        print("Reservation Prediction is finished")
        print('')

    ####################################
    # 2. Data & Variable Initialization
    ####################################
    def _load_data(self, type_apply: str):
        applied_list = self.type_apply[type_apply]
        data_map = defaultdict(dict)
        for data_type in self.type_data:    # cnt / disc / util
            for model in applied_list:
                data_type_name = self.type_data_map[data_type]
                data_map[data_type].update({model: pd.read_csv(os.path.join(self.path_data_hx, type_apply,
                                                               data_type_name, data_type_name + '_' + model + '.csv'))})

        return data_map

    ##################################
    # 2. Data Pre-processing
    ##################################
    def _split_input_target_all(self, applied_list: list):
        io = {}
        for data_type in self.type_data:
            io_model = {}
            for model in applied_list:
                split = self._split_to_input_target(data_type=data_type, model=model)
                io_model[model] = split
            io[data_type] = io_model

        return io

    def _split_to_input_target(self, data_type: str, model: str):
        x = self.data_map[data_type][model].drop(columns=self.split_map[data_type]['drop'])
        y = self.data_map[data_type][model][self.split_map[data_type]['target']]

        return {'x': x, 'y': y}

    def _split_train_test_all(self, data: dict):
        result = {}
        for type_key, type_val in data.items():    # data_type: cnt / disc / util
            model_dict = {}
            for model_key, model_val in type_val.items():    # model: av / k3 / vl / su
                train, test = self._split_train_test(x=model_val['x'], y=model_val['y'])
                model_dict[model_key] = {'train': train, 'test': test}
            result[type_key] = model_dict
            # result[data_type] = {model: {'train': train, 'test': test}}

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
            for model_key, model_val in type_val.items():    # model_key: av / k3 / vl / su
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

    def _save_best_params(self, regr: str, regr_bests: dict, type_apply: str):
        for type_key, type_val in regr_bests.items():
            for model_key, model_val in type_val.items():
                best_params = model_val.get_params()
                f = open(os.path.join(self.path_res_pred, type_apply, type_key,
                                      regr + '_params_' + model_key + '.pickle'), 'wb')
                pickle.dump(best_params, f)
                f.close()

    ##################################
    # 4. Prediction
    ##################################
    def _pred(self, pred_day: str, fitted_model: dict, applied_list: list):
        # Get season value and initial discount rate
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        season = self.season_re[pred_datetime]
        init_disc = self.disc_last_week[pred_datetime]
        init_capa = self.capacity_re[pred_datetime]

        # Make initial values dataframe
        pred_input = self._get_pred_input(pred_day=pred_day, season=season, init_disc=init_disc,
                                          applied_list=applied_list)

        pred_result = self._pred_fitted_model(pred_datetime=pred_datetime, pred_input=pred_input,
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

    def _get_pred_input(self, pred_day: str, season: int, init_disc: dict, applied_list: list):
        pred_datetime = dt.datetime.strptime(pred_day, '%Y-%m-%d')
        pred_input = {}
        for data_type in self.type_data:
            input_model = {}
            for model in applied_list:
                if data_type in ['cnt', 'util']:
                    input_model[model] = pd.DataFrame({'season': season, 'lead_time': self.lt_vec,
                                                       'discount': init_disc[model]})
                else:
                    input_model[model] = pd.DataFrame({'season': season, 'lead_time': self.lt_vec,
                                                       'res_cnt': self.res_rate_re[pred_datetime].get(model, 0)})
            pred_input[data_type] = input_model

        return pred_input

    def _set_recent_dataset(self, disc_confirm_last_week: str, type_apply: str):
        self.capacity_re = self._get_capacity(time='re', type_apply=type_apply)
        self.res_rate_re = self._get_res_rate(time='re', type_apply=type_apply)
        self.season_re = self._get_season_map(time='re')
        self.disc_last_week = self._get_disc_last_week(disc_confirm_last_week=disc_confirm_last_week)
        self.lt, self.lt_vec, self.lt_to_lt_vec = self.utility.get_lead_time()

    def _get_res_rate(self, time: str, type_apply: str):
        # Load recent reservation dataset
        res_re = self.utility.load_res(time=time, status_update_day=self.res_status_ud_day)
        res_re = res_re.rename(columns=self.utility.RENAME_COL_RES)

        # Drop unnecessary columns
        res_drop_col = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm', 'tot_fee',
                        'res_model', 'car_grd', 'rent_time', 'return_day', 'return_time', 'rent_period_day',
                        'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm', 'sale_purpose',
                        'applied_discount', 'discount_rate', 'member_grd', 'sale_purpose', 'car_kind']
        res_re = res_re.drop(columns=res_drop_col, errors='ignore')

        res_re['rent_day'] = pd.to_datetime(res_re['rent_day'], format='%Y-%m-%d')
        res_re['res_day'] = pd.to_datetime(res_re['res_day'], format='%Y-%m-%d')

        # filter only 1.6 grade car group
        res_re = self.utility.filter_model_grade(df=res_re)

        # Car Model group
        res_re = self.utility.cluster_model(df=res_re, type_apply=type_apply)
        res_re = res_re.drop(columns=['res_model_nm'], errors='ignore')
        res_re = res_re.sort_values(by=['rent_day', 'res_day'])

        res_cnt = res_re.groupby(by=['rent_day', 'res_model']).count()['res_num']
        res_cnt = res_cnt.reset_index(level=(0, 1))

        res_cnt = self._add_capacity(df=res_cnt)
        res_cnt['res_rate'] = res_cnt['res_num'] / res_cnt['capa']

        res_curr_dict = defaultdict(dict)
        for day, model, cnt in zip(res_cnt['rent_day'], res_cnt['res_model'], res_cnt['res_rate']):
            res_curr_dict[day].update({model: cnt})

        return res_curr_dict

    def _add_capacity(self, df: pd.DataFrame):
        df['capa'] = df[['rent_day', 'res_model']].apply(self._set_capa, axis=1)

        return df

    def _set_capa(self, x):
        return self.capacity_re[x[0]][x[1]]

    def _get_season_map(self, time: str):
        # Load recent seasonality dataset
        season_re = self.utility.load_season(time=time)
        day_to_season = {day: season for day, season in zip(season_re['rent_day'], season_re['seasonality'])}

        return day_to_season

    def _get_disc_last_week(self, disc_confirm_last_week: str):
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'disc_confirm')
        disc_confirm = pd.read_csv(os.path.join(load_path, 'disc_confirm_' + disc_confirm_last_week + '.csv'),
                                   delimiter='\t', dtype={'date': str, 'model': str, 'disc': int})
        disc_confirm['date'] = pd.to_datetime(disc_confirm['date'], format='%Y%m%d')

        disc_last_week = defaultdict(dict)
        for date, model, disc in zip(disc_confirm['date'], disc_confirm['model'], disc_confirm['disc']):
            disc_last_week[date].update({self.model_nm_map[model]: disc})

        return disc_last_week

    def _get_capacity(self, time: str, type_apply: str):
        # Load capacity of car models
        capa_re = self.utility.load_capacity(time=time, type_apply=type_apply)
        # Load unavailable capacity of car models
        capa_re_unavail = self.utility.load_capacity(time='re', type_apply=type_apply, unavail=True)
        capa_re_unavail = capa_re_unavail.rename(columns={'capa': 'unavail'})
        # Convert monthly capacity to daily
        capa_re = self.utility.conv_mon_to_day(df=capa_re)
        # Subtract unavailable capacity
        capa_re = self.utility.apply_unavail_capa(capacity=capa_re, capa_unavail=capa_re_unavail)
        # Mapping dictionary: (Data, Model) -> capacity
        capa_re_dict = defaultdict(dict)
        for date, model, capa in zip(capa_re['date'], capa_re['model'], capa_re['capa']):
            capa_re_dict[date].update({self.model_nm_map[model]: capa - self.avg_unavail_capa})

        return capa_re_dict

    def _load_best_params(self, regr: str, type_apply: str):
        detail_type = self.type_apply[type_apply]
        regr_bests = {}
        for data_type in self.type_data:
            model_bests = {}
            for model in detail_type:
                f = open(os.path.join(self.path_res_pred, type_apply, data_type,
                                      regr + '_params_' + model + '.pickle'), 'rb')
                model_bests[model] = pickle.load(f)
                f.close()
            regr_bests[data_type] = model_bests

        return regr_bests

    def _fit_model(self, dataset: dict, regressor: str, params: dict):
        fitted = {}
        for type_key, type_val in params.items():
            model_fit = {}
            for model_key, model_val in type_val.items():
                model = self.REGRESSORS[regressor](**model_val)
                data = dataset[type_key][model_key]
                model.fit(data['x'], data['y'])
                model_fit[model_key] = model
            fitted[type_key] = model_fit

        return fitted

    def _pred_fitted_model(self, pred_datetime, pred_input: dict, fitted_model: dict):
        pred_results = {}
        for type_key, type_val in fitted_model.items():
            pred_models = {}
            for model_key, model_val in type_val.items():
                prediction = model_val.predict(pred_input[type_key][model_key])
                if type_key == 'cnt':
                    capacity = self.capacity_re[pred_datetime][model_key]
                    prediction = np.round(prediction * capacity, 1)
                else:
                    prediction = np.round(prediction, 3)
                pred_models[model_key] = prediction
            pred_results[type_key] = pred_models

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
            for model_key, model_val in type_val.items():   # model_key: av / k3 / vl / su
                rslt = [model_val[self.lt_to_lt_vec[i]] for i in self.lt]
                result[model_key].update({type_key: rslt})

        return result

    def _conv_to_dataframe(self, result: dict, pred_datetime: dt.datetime,
                           init_disc: dict, init_capa: dict):
        date = [pred_datetime - dt.timedelta(days=int(i*-1)) for i in self.lt]
        lead_time = self.lt

        model_df = {}
        for model_key, model_val in result.items():
            df = pd.DataFrame({'date': date, 'lead_time': lead_time, 'curr_disc': init_disc[model_key]})
            for type_key, type_val in model_val.items():
                df['exp_' + type_key] = type_val

            # Calculate utilization count
            df['exp_util_cnt'] = df['exp_util'] * init_capa[model_key]
            model_df[model_key] = df

        return model_df

    def _save_result(self, result: dict, pred_day: str):
        save_path = os.path.join('..', 'result', 'data', 'prediction', self.apply_day)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for model_key, model_val in result.items():
            save_model_path = os.path.join(save_path, model_key)
            if not os.path.exists(save_model_path):
                os.mkdir(save_model_path)
            model_val.to_csv(os.path.join(save_model_path, 'res_pred(' + pred_day + ').csv'), index=False)
