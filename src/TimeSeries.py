import os
import math
import pickle
import datetime as dt
from itertools import product

import pandas as pd

from sklearn.metrics import mean_squared_error

# Time Series models
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeries(object):
    """
    Jeju visitors prediction model (Time series model)
    """

    def __init__(self, start_date: str, end_date: str):
        self.load_path = os.path.join('..', 'input', 'demand')
        self.save_path = os.path.join('..', 'result', 'model', 'time_series')
        self.start_date = start_date
        self.end_date = end_date
        self.model: dict = {'ar': self.model_ar,
                            'arima': self.model_arima,
                            'hw': self.model_hw}

    def _filter_period(self, df: pd.DataFrame):
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        start_date = dt.datetime.strptime(self.start_date, '%Y%m%d')
        end_date = dt.datetime.strptime(self.end_date, '%Y%m%d')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        return df

    def train(self, n_test: int, test_models: list, param_grids: dict):
        visitor = pd.read_csv(os.path.join(self.load_path, 'jeju_visit_daily.csv'), delimiter='\t')
        visitor = visitor.rename(columns={'yyyymmdd': 'date'})

        # Filter date
        data = self._filter_period(df=visitor)

        # String date convert to datetime
        data = data.set_index('date')

        # Split dataset to domestic and foreign
        jeju_dom = data[['domestic']]
        jeju_for = data[['foreign']]

        # Walk Forward Validation
        # Domestic
        err_dom = []
        for model in test_models:
            err_dom.append([model, self.tune_hyper_parameter(model=model, data=jeju_dom,
                                                             n_test=n_test, param_grid=param_grids[model])])
        best_param_dom = sorted(err_dom, key=lambda x: x[1][1])[0]

        # Foreign
        err_for = []
        for model in test_models:
            err_for.append([model, self.tune_hyper_parameter(model=model, data=jeju_for,
                                                             n_test=n_test, param_grid=param_grids[model])])
        best_param_for = sorted(err_for, key=lambda x: x[1][1])[0]

        # Save best model for each group(domestic / foreign)
        for visit, param_best in zip(['dom', 'for'], [best_param_dom, best_param_for]):
            best_params = {key: val for key, val in param_best[1][0]}
            best_model_params = {param_best[0]: best_params}
            f = open(os.path.join(self.save_path, ''.join(['jeju_', visit, '_best_params.pickle'])), 'wb')
            pickle.dump(best_model_params, f)
            f.close()

        print("Training of demand prediction model is finished")

    def predict(self, pred_step: int):
        # Load best parameters for model
        best_params = {}
        for visit in ['dom', 'for']:
            f = open(os.path.join(self.save_path, ''.join(['jeju_', visit, '_best_params.pickle'])), 'rb')
            best_params[visit] = pickle.load(f)
            f.close()

        # Load jeju visitor dataset
        visitor = pd.read_csv(os.path.join(self.load_path, 'jeju_visit_daily.csv'), delimiter='\t')
        visitor = visitor.rename(columns={'yyyymmdd': 'date'})
        data = self._filter_period(df=visitor)

        # String date convert to datetime
        data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
        data = data.set_index('date')

        # Split dataset to domestic and foreign
        jeju_dom = data[['domestic']]
        jeju_for = data[['foreign']]

        pred_dom = self.model[list(best_params['dom'].keys())[0]](data=jeju_dom,
                                                                  config=list(best_params['dom'].values())[0],
                                                                  pred_step=pred_step + 1)
        pred_for = self.model[list(best_params['for'].keys())[0]](data=jeju_for,
                                                                  config=list(best_params['for'].values())[0],
                                                                  pred_step=pred_step + 1)
        pred_tot = pred_dom + pred_for

        # Compare
        visit_19_12 = data.loc[dt.datetime(2019, 12, 1):dt.datetime(2019, 12, 31)]['total']
        visit_20_01 = data.loc[dt.datetime(2020, 1, 1):dt.datetime(2020, 1, 31)]['total']
        visit_20_02 = data.loc[dt.datetime(2020, 2, 1):dt.datetime(2020, 2, 28)]['total']

        pred_20_12 = pred_tot.loc[dt.datetime(2020, 12, 1):dt.datetime(2020, 12, 31)]
        pred_21_01 = pred_tot.loc[dt.datetime(2021, 1, 1):dt.datetime(2021, 1, 31)]
        pred_21_02 = pred_tot.loc[dt.datetime(2021, 2, 1):dt.datetime(2021, 2, 28)]

        chg_rate_20_12 = round((pred_20_12.mean() / visit_19_12.mean() - 1), 2)
        chg_rate_21_01 = round((pred_21_01.mean() / visit_20_01.mean() - 1), 2)
        chg_rate_21_02 = round((pred_21_02.mean() / visit_20_02.mean() - 1), 2)

        dmd_pred = pd.DataFrame({'date': ['202012', '202101', '202102'],
                                 'dmd_chg': [chg_rate_20_12, chg_rate_21_01, chg_rate_21_02]})
        dmd_pred.to_csv(os.path.join(self.save_path, 'dmd_pred_2012_2102.csv'), index=False)

        print("Model prediction is finished")

    # Time series model
    @staticmethod
    def model_ar(data: pd.DataFrame, config: dict, pred_step=1):
        """
        AR model
        :param data: times series data
        :param config:
                 - lags: the number of lags
                 - trend: the trend to include in the model
                        n: No Trend
                        c: Constant Only
                        t: Time Trend Only
                        ct: Constant and time trend
                 - seasonal: Flag indicating whether to include seasonal dummies in the model
        :param pred_step: prediction steps
        """
        model = AutoReg(endog=data, lags=config['lags'], trend=config['trend'])
        model_fit = model.fit()
        prediction = model_fit.predict(start=len(data), end=len(data) + pred_step - 1)

        return prediction

    @staticmethod
    def model_arima(data: pd.DataFrame, config: dict, pred_step=1):
        """
        ARIMA model
        :param data: time series data
        :param config:
             - order: (p, d, q)
                    p: Trend auto-regression order
                    d: Trend difference order
                    q: Trend moving average order
             - trend: the trend to include in the model
                    n: No Trend
                    c: Constant
                    t: Trend
                    ct: Constant and Trend
         :param pred_step: prediction steps
        """
        model = ARIMA(endog=data, order=config['order'], trend=config['trend'])
        model_fit = model.fit()
        prediction = model_fit.predict(start=len(data), end=len(data) + pred_step - 1)

        return prediction

    @staticmethod
    def model_hw(data: pd.DataFrame, config: dict, pred_step=1):
        """
        Holt-winters model
        :param data: time series data
        :param config:
                - trend - type of trend component
                    ('add', 'mul', 'additive', 'multiplicative')
                - damped_trend - should the trend component be damped
                - seasonal - Type of seasonal component
                        ('add', 'mul', 'additive', 'multiplicative', None)
                - seasonal_periods - The number of periods in a complete seasonal cycle
        :param pred_step: prediction steps
        """
        model = ExponentialSmoothing(endog=data, trend=config['trend'], damped_trend=config['damped_trend'])
        model_fit = model.fit()
        prediction = model_fit.predict(start=len(data), end=len(data) + pred_step - 1)

        return prediction

    def validate_walk_forward(self, model: str, data: pd.DataFrame, n_test: int, config: dict):
        # Split dataset
        train, test = data[:-n_test], data[-n_test:]

        # Validation
        print(f'Model: {model}, Configuration: {config}')
        prediction = self.model[model](data=train, config=config, pred_step=n_test)

        # Add actual observation to history for the next loop
        error = math.sqrt(mean_squared_error(test.values, prediction))
        print(f'Error: {error}')

        return error

    def tune_hyper_parameter(self, model: str, data: pd.DataFrame, n_test: int, param_grid: dict):
        err_list = list()
        for params in list(product(*list(param_grid.values()))):
            config = {key: val for key, val in zip(list(param_grid.keys()), params)}
            err = self.validate_walk_forward(model=model, data=data, n_test=n_test, config=config)
            err_list.append((list(config.items()), round(err, 2)))

        err_list = sorted(err_list, key=lambda x: x[1])

        return err_list[0]
