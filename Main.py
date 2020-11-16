from Model1 import MODEL1
from Model2 import MODEL2
from Model3 import MODEL3

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# Model 1 : Jeju visitor prediction

def run_model_1():
    # Define hyperparameters
    start_date = 20191101
    end_date = 20201031
    n_test = 28     # 4 weeks
    test_models = ['ar', 'hw']      # ar / Holt-winters

    # Parameters Grids
    param_grids = {
        'ar': {
            'lags': list(np.arange(1, 15, 1)),
            'trend': ['c', 't', 'ct']},
        'hw': {
            'trend': ['add', 'additive'],
            'damped_trend': [True, False]}}

    jeju_visitors = pd.read_csv(os.path.join('..', 'input', 'jeju_visit_daily.csv'), delimiter='\t')

    model_1 = MODEL1(visitor=jeju_visitors, start_date=20191101, end_date=20201031)

    # Train
    model_1.training(n_test=n_test, test_models=test_models, param_grids=param_grids)

    # Prediction
    model_1.prediction(pred_step=88+31)

# Model 2
def run_model_2():

    


    pass

# Moin

# run_model_1()
run_model_2()