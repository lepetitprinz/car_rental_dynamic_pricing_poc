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

    def __init__(self, res_update_day: str):
        # Path of data & model
        self.path_data_hx = os.path.join('..', 'result', 'data', 'model_2', 'hx')
        self.path_model = os.path.join('..', 'result', 'model', 'model_2')

        self.random_state = 2020
        self.test_size = 0.2
        self.data_type: list = ['cnt', 'disc', 'util']
        self.data_type_map: dict = {'cnt': 'cnt_cum', 'disc': 'cnt_cum', 'util': 'util_cum'}
        self.model_type: list = ['av', 'k3', 'vl', 'su']
        self.car_type: list = ['av_ad', 'av_new', 'k3', 'soul', 'vlst']

        # Prediction variables
        # Initial values of variables
        self.res_update_day = res_update_day
        self.res_cnt_re: dict = {}
        self.season_re: dict = {}
        self.disc_re: dict = {}
        self.capa_re: dict = {}
        self.avg_unavail_capa = 2

        # Lead time
        self.lt: np.array = []
        self.lt_vec: np.array = []
        self.lt_to_lt_vec: dict = {}

        # inintialize mapping dictionary
        self.data_map: dict = dict()
        self.split_map: dict = dict()
        self.param_grids = dict()

    def train(self, model_detail: str):
        # Load dataset
        self.data_map = self._load_data(model_detail=model_detail)

        # Define input and output
        self.split_map = self._set_split_map()

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

    def predict(self, pred_days: list, model_detail: str):
        # Load dataset
        self.data_map = self._load_data(model_detail=model_detail)

        # Define input and output
        self.split_map = self._set_split_map()

        # Split into input and output
        m2_io = self._split_input_target_all()

        # Set initial variables
        self._set_recent_dataset()

        # Load best hyper-parameters
        extr_bests = self._load_best_params(regr='extr')

        # fit the model
        fitted = self._fit_model(dataset=m2_io, regr='extr', params=extr_bests)

        for pred_day in pred_days:
            self._pred(pred_day=pred_day, fitted_model=fitted)

        print('')
        print("Model 2 Prediction is finished")
        print('')

    ####################################
    # 2. Data & Variable Initialization
    ####################################
    def _load_data(self, model_detail: str):
        data_map = defaultdict(dict)
        if model_detail == 'model':
            for data_type in self.data_type:        # cnt / disc / util
                for model in self.model_type:       # av / k3 / su / vl
                    data_type_name = self.data_type_map[data_type]
                    data_map[data_type].update({model: pd.read_csv(os.path.join(self.path_data_hx, model_detail,
                                                data_type_name, data_type_name + '_' + model + '.csv'))})
        elif model_detail == 'car':
            for data_type in self.data_type:    # cnt / disc / util
                for model in self.car_type:     # av_ad / av_new / k3 / soul / vlst
                    data_type_name = self.data_type_map[data_type]
                    data_map[data_type].update({model: pd.read_csv(os.path.join(self.path_data_hx, model_detail,
                                                data_type_name, data_type_name + '_' + model + '.csv'))})

        return data_map

    @staticmethod
    def _set_split_map():
        split_map = {'cnt': {'drop': ['cnt_cum'],
                             'target': 'cnt_cum'},
                     'disc': {'drop': ['disc_mean'],
                              'target': 'disc_mean'},
                     'util': {'drop': ['util_cum', 'util_rate_cum'],
                              'target': 'util_rate_cum'}}

        return split_map

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
    def _pred(self, pred_day: str, fitted_model: dict):
        # Get season value and initial discount rate
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        season = self.season_re[pred_datetime]
        init_disc = self.disc_re[pred_datetime]

        # Set initial capacity of model
        init_capa = {'av': self.capa_re[(pred_datetime, 'AVANTE')] - self.avg_unavail_capa,
                     'k3': self.capa_re[(pred_datetime, 'K3')] - self.avg_unavail_capa,
                     'vl': self.capa_re[(pred_datetime, 'VELOSTER')] - self.avg_unavail_capa,
                     'su': self.capa_re[(pred_datetime, 'SOUL')] - self.avg_unavail_capa}

        # Make initial values dataframe
        pred_input = self._get_pred_input(season=season, init_disc=init_disc, pred_day=pred_day)

        pred_result = self._pred_fitted_model(pred_input=pred_input, fitted_model=fitted_model)

        # Map lead time to prediction results
        lt_to_pred_result = self._get_lt_to_pred_result(pred_result=pred_result)

        result = self._map_rslt_to_lead_time(pred_final=lt_to_pred_result)

        # Result data convert to dataframe
        result_df = self._conv_to_dataframe(result=result, pred_datetime=pred_datetime,
                                            init_disc=init_disc, init_capa=init_capa)

        # Save the result dataframe
        self._save_result(result=result_df, pred_day=pred_day)

        print(f'Prediction result on {pred_day} is saved')

    def _get_pred_input(self, season: int, init_disc: int, pred_day: str):
        pred_input = {}
        for data_type in self.data_type:
            input_model = {}
            for model in self.model_type:
                if data_type in ['cnt', 'util']:
                    input_model[model] = pd.DataFrame({'season': season,
                                                       'lead_time': self.lt_vec,
                                                       'discount': init_disc})
                else:
                    input_model[model] = pd.DataFrame({'season': season, 'lead_time': self.lt_vec,
                                                       'res_cnt': self.res_cnt_re[pred_day].get(model, 0)})
            pred_input[data_type] = input_model

        return pred_input

    def _set_recent_dataset(self):
        self.res_cnt_re = self._get_res_cnt_re()
        self.season_re = self._get_seasonal_map()
        self.disc_re = self._get_disc_re()
        self.capa_re = self._get_capa_re()
        self.lt, self.lt_vec, self.lt_to_lt_vec = self._get_lead_time()

    def _get_res_cnt_re(self):
        # Load recent reservation dataset
        load_path = os.path.join('..', 'input', 'reservation')
        res_re = pd.read_csv(os.path.join(load_path, 'res_' + self.res_update_day + '.csv'), delimiter='\t')

        res_remap_cols = {
            '예약경로': 'res_route', '예약경로명': 'res_route_nm', '계약번호': 'res_num',
            '고객구분': 'cust_kind', '고객구분명': 'cust_kind_nm', '총 청구액(VAT포함)': 'tot_fee',
            '예약모델': 'res_model', '예약모델명': 'res_model_nm', '차급': 'car_grd',
            '대여일': 'rent_day', '대여시간': 'rent_time', '반납일': 'return_day', '반납시간': 'return_time',
            '대여기간(일)': 'rent_period_day', '대여기간(시간)': 'rent_period_time',
            'CDW요금': 'cdw_fee', '할인유형': 'discount_type', '할인유형명': 'discount_type_nm',
            '적용할인명': 'applyed_discount', '적용할인율(%)': 'discount_rate', '회원등급': 'member_grd',
            '구매목적': 'sale_purpose', '생성일': 'res_day', '차종': 'car_kind'}
        res_re = res_re.rename(columns=res_remap_cols)

        res_drop_col = ['res_route', 'res_route_nm', 'cust_kind', 'cust_kind_nm', 'tot_fee',
                        'res_model', 'car_grd', 'rent_time', 'return_day', 'return_time', 'rent_period_day',
                        'rent_period_time', 'cdw_fee', 'discount_type', 'discount_type_nm', 'sale_purpose',
                        'applyed_discount', 'discount_rate', 'member_grd', 'sale_purpose', 'car_kind']
        res_re = res_re.drop(columns=res_drop_col, errors='ignore')

        res_re['rent_day'] = pd.to_datetime(res_re['rent_day'], format='%Y-%m-%d')
        res_re['res_day'] = pd.to_datetime(res_re['res_day'], format='%Y-%m-%d')

        # filter only 1.6 grade car group
        res_re = res_re[res_re['res_model_nm'].isin([
            'ALL NEW K3 (G)',
            '아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)',
            '더 올 뉴 벨로스터 (G)', '쏘울 (G)', '쏘울 부스터 (G)'
        ])]

        # Car Model group
        # SOUL 모델은 VELOSTER 모델에 포함해서 분석 (실적 데이터가 적음)
        conditions = [
            res_re['res_model_nm'].isin(['ALL NEW K3 (G)']),
            res_re['res_model_nm'].isin(['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)']),
            res_re['res_model_nm'].isin(['더 올 뉴 벨로스터 (G)']),
            res_re['res_model_nm'].isin(['쏘울 (G)', '쏘울 부스터 (G)'])
        ]
        values = ['k3', 'av', 'vl', 'su']
        res_re['res_model_grp'] = np.select(conditions, values)

        res_re = res_re.drop(columns=['res_model_nm'], errors='ignore')
        res_re = res_re.sort_values(by=['rent_day', 'res_day'])

        res_cnt = res_re.groupby(by=['rent_day', 'res_model_grp']).count()['res_num']
        res_cnt = res_cnt.reset_index(level=(0, 1))

        res_curr_dict = defaultdict(dict)
        for day, model, cnt in zip(res_cnt['rent_day'], res_cnt['res_model_grp'], res_cnt['res_num']):
            day_str = day.strftime('%Y-%m-%d')
            res_curr_dict[day_str].update({model: cnt})

        return res_curr_dict

    @staticmethod
    def _get_seasonal_map():
        # Load recent seasonality dataset
        load_path = os.path.join('..', 'input', 'seasonality')
        ss_curr = pd.read_csv(os.path.join(load_path, 'seasonality_curr.csv'), delimiter='\t')
        ss_curr['date'] = pd.to_datetime(ss_curr['date'], format='%Y%m%d')
        day_to_season = {day: season for day, season in zip(ss_curr['date'], ss_curr['seasonality'])}

        return day_to_season

    @staticmethod
    def _get_disc_re():
        # Initial discount rate for each season
        load_path = os.path.join('..', 'input', 'discount')
        dscnt_init = pd.read_csv(os.path.join(load_path, 'discount_init.csv'), delimiter='\t')
        dscnt_init['date'] = pd.to_datetime(dscnt_init['date'], format='%Y%m%d')
        day_to_init_discount = {day: discount for day, discount in zip(dscnt_init['date'],
                                                                       dscnt_init['discount_init'])}

        return day_to_init_discount

    def _get_capa_re(self):
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'capa')
        capa_init = pd.read_csv(os.path.join(load_path, 'capa_curr_model.csv'), delimiter='\t',
                                dtype={'date': str, 'model': str, 'capa': int})
        capa_init_unavail = pd.read_csv(os.path.join(load_path, 'capa_unavail_model.csv'), delimiter='\t')

        capa_init = self._conv_mon_to_day(df=capa_init, end_day='28')
        capa_init = self._apply_unavail_capa(capa=capa_init, capa_unavail=capa_init_unavail)

        capa_init_dict = {(date, model): capa for date, model, capa in zip(capa_init['date'],
                                                                           capa_init['model'],
                                                                           capa_init['capa'])}

        return capa_init_dict

    @staticmethod
    def _conv_mon_to_day(df: pd.DataFrame, end_day: str):
        months = np.sort(df['date'].unique())
        days = pd.date_range(start=months[0] + '01', end=months[-1] + end_day)
        model_unique = df[['model', 'capa']].drop_duplicates()

        df_days = pd.DataFrame()
        for model, capa in zip(model_unique['model'], model_unique['capa']):
            temp = pd.DataFrame({'date': days, 'model': model, 'capa': capa})
            df_days = pd.concat([df_days, temp], axis=0)

        return df_days

    @staticmethod
    def _apply_unavail_capa(capa: pd.DataFrame, capa_unavail: pd.DataFrame):
        capa_unavail['date'] = pd.to_datetime(capa_unavail['date'], format='%Y%m%d')
        capa_new = pd.merge(capa, capa_unavail, how='left', on=['date', 'model'], left_index=True, right_index=False)
        capa_new = capa_new.fillna(0)
        capa_new['capa'] = capa_new['capa'] - capa_new['unavail']

        return capa_new

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
    def _pred_fitted_model(pred_input: dict, fitted_model: dict):
        pred_results = {}
        for type_key, type_val in fitted_model.items():
            pred_models = {}
            for model_key, model_val in type_val.items():
                pred = model_val.predict(pred_input[type_key][model_key])
                if (type_key == 'cnt') or (type_key == 'disc'):
                    pred = np.round(pred, 1)
                else:
                    pred = np.round(pred, 3)
                pred_models[model_key] = pred
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
            model_val.to_csv(os.path.join(save_path, model_key,
                                          'm2_pred(' + pred_day + ').csv'), index=False)
