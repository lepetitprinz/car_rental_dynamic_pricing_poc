import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import ExtraTreesRegressor

class MODEL2(object):

    REGRESSORS = {"Extra Trees Regressor": ExtraTreesRegressor(),
                  "extr": ExtraTreesRegressor}

    def __init__(self):
        self.load_path = os.path.join('..', 'result', 'data', 'model_2')
        self.load_model_path = os.path.join('..', 'result', 'model', 'model_2')
        self.save_path = os.path.join('..', 'result', 'model', 'model_2')
        self.random_state = 2020
        self.test_size = 0.2

        # Load reservation count dataset
        self.res_cnt_av: pd.DataFrame = pd.DataFrame()
        self.res_cnt_k3: pd.DataFrame = pd.DataFrame()
        self.res_cnt_vl: pd.DataFrame = pd.DataFrame()
        # Load reservation utilization dataset
        self.res_util_av: pd.DataFrame = pd.DataFrame()
        self.res_util_k3: pd.DataFrame = pd.DataFrame()
        self.res_util_vl: pd.DataFrame = pd.DataFrame()
        # inintialize mapping dictionary
        self.data_map: dict = dict()
        self.split_map: dict = dict()
        self.param_grids = dict()

        #
        self._load_data()
        self.data_map, self.split_map = self._set_split_map()

        # Prediction variables
        # Initial values of variables
        self.day_to_season: dict = {}
        self.day_to_init_disc: dict = {}
        self.mon_to_init_capa: dict = {}
        # Lead time
        self.lt: np.array = []
        self.lt_vec: np.array = []
        self.lt_to_lt_vec: dict = {}

    def train(self):
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

    def predict(self):
        # Split into input and output
        m2_io = self._split_input_target_all()

        # Set initial variables
        self._set_pred_init_variables()

        # Load best hyper-parameters
        extr_bests = self._load_best_params(regr='extr')

        # fit the model
        fitted = self._fit_model(dataset=m2_io, regr='extr', params=extr_bests)

        pred_day = '2020-12-18'

        lt_to_pred_result = self._pred(pred_day=pred_day, fitted_model=fitted)

    def _pred(self, pred_day: str, fitted_model: dict):
        # Get season value and initial discount rate
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        season = self.day_to_season[pred_datetime]
        init_disc = self.day_to_init_disc[pred_datetime]

        # Set initial capacity of model
        pred_mon = pred_day.split('-')[0] + pred_day.split('-')[1]
        init_capa = {'av': self.mon_to_init_capa[(pred_mon, 'AVANTE')],
                     'k3': self.mon_to_init_capa[(pred_mon, 'AVANTE')],
                     'vl': self.mon_to_init_capa[(pred_mon, 'AVANTE')]}

        # Make initial values dataframe
        pred_input = pd.DataFrame({'season': season, 'lead_time': self.lt_vec, 'discount': init_disc})

        pred_result = self._pred_fitted_model(pred_input=pred_input, fitted_model=fitted_model)

        # Map lead time to prediction results
        lt_to_pred_result = self._get_lt_to_pred_result(pred_result=pred_result)

        return lt_to_pred_result

    ####################################
    # 2. Data & Variable Initialization
    ####################################
    def _load_data(self):
        # Load Reservation Count dataset
        self.res_cnt_av = pd.read_csv(os.path.join(self.load_path, 'model_2_cnt_av.csv'))
        self.res_cnt_k3 = pd.read_csv(os.path.join(self.load_path, 'model_2_cnt_k3.csv'))
        self.res_cnt_vl = pd.read_csv(os.path.join(self.load_path, 'model_2_cnt_vl.csv'))
        # Load Reservation Utilization dataset
        self.res_util_av = pd.read_csv(os.path.join(self.load_path, 'model_2_util_av.csv'))
        self.res_util_k3 = pd.read_csv(os.path.join(self.load_path, 'model_2_util_k3.csv'))
        self.res_util_vl = pd.read_csv(os.path.join(self.load_path, 'model_2_util_vl.csv'))

    def _set_split_map(self):
        data_map = {'cnt': {'av': self.res_cnt_av,
                            'k3': self.res_cnt_k3,
                            'vl': self.res_cnt_vl},
                    'disc': {'av': self.res_cnt_av,
                             'k3': self.res_cnt_k3,
                             'vl': self.res_cnt_vl},
                    'util': {'av': self.res_util_av,
                             'k3': self.res_util_k3,
                             'vl': self.res_util_vl}}

        split_map = {'cnt': {'drop': ['cum_cnt'],
                             'target': 'cum_cnt'},
                     'disc': {'drop': ['cum_dscnt_mean'],
                              'target': 'cum_dscnt_mean'},
                     'util': {
                         'drop': ['cum_util_time', 'cum_util_cnt', 'capa', 'cum_util_time_rate', 'cum_util_cnt_rate'],
                         'target': 'cum_util_time_rate'}}

        return data_map, split_map

    ##################################
    # 2. Data Preprcessing
    ##################################
    def _split_input_target_all(self):
        io = {}
        for data_type in ['cnt', 'disc', 'util']:
            io_model = {}
            for model in ['av', 'k3', 'vl']:
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
            for model_key, model_val in type_val.items():    # model: av / k3 / vl
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
                f = open(os.path.join(self.save_path, type_key + '_' + model_key + '_' + regr + '_params.pickle'), 'wb')
                pickle.dump(best_params, f)
                f.close()

    ##################################
    # 4. Prediction
    ##################################
    def _set_pred_init_variables(self):
        self.day_to_season = self._get_seasonal_map()
        self.day_to_init_disc = self._get_init_disc_map()
        self.mon_to_init_capa = self._get_init_capa()
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
        init_capa = pd.read_csv(os.path.join(load_path, 'capa_curr.csv'), delimiter='\t',
                                dtype={'date': str, 'model': str, 'capa': int})
        day_to_init_capa = {(date, model): capa for date, model, capa in zip(init_capa['date'],
                                                                             init_capa['model'],
                                                                             init_capa['capa'])}

        return day_to_init_capa

    @staticmethod
    def _get_lead_time():
        # Lead Time Setting
        lt = np.arange(-83, 1, 1)
        lt_vec = np.arange(-36, 1, 1)

        lt_to_lt_vec = {-1 * i: (((i // 7) + 24) * -1 if i > 28 else i * -1) for i in range(0, 84, 1)}

        return lt, lt_vec, lt_to_lt_vec

    def _load_best_params(self, regr: str):
        regr_bests = {}
        for data_type in ['cnt', 'disc', 'util']:
            model_bests = {}
            for model in ['av', 'k3', 'vl']:
                f = open(os.path.join(self.load_model_path, data_type + '_' + model + '_' + regr + '_params.pickle'), 'rb')
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

    def _pred_fitted_model(self, pred_input: pd.DataFrame, fitted_model: dict):
        pred_results = {}
        for type_key, type_val in fitted_model.items():
            pred_models = {}
            for model_key, model_val in type_val.items():
                pred = model_val.predict(pred_input)
                pred_models[model_key] = np.round(pred, 3)
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
