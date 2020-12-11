import os
import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


class Utility(object):

    def __init__(self):
        self.grade_1_6 = ['아반떼 AD (G)', '아반떼 AD (G) F/L', '올 뉴 아반떼 (G)',
                          'ALL NEW K3 (G)',
                          '쏘울 (G)', '쏘울 부스터 (G)', '더 올 뉴 벨로스터 (G)']

    @staticmethod
    def get_season(time: str):
        """
        :param time: hx / re (history or recent)
        :return:
        """
        # Load seasonal dataset
        load_path = os.path.join('..', 'input', 'seasonality', 'seasonality_' + time + '.csv')
        season = pd.read_csv(load_path, delimiter='\t')
        season['rent_day'] = pd.to_datetime(season['rent_day'], format='%Y-%m-%d')

        return season



    @staticmethod
    def _get_res_util(df: pd.DataFrame):
        res_util = []
        for rent_d, rent_t, return_d, return_t, res_day, discount, model in zip(
                df['rent_day'], df['rent_time'], df['return_day'], df['return_time'],
                df['res_day'], df['discount'], df['res_model']):

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
                date_range, [res_day] * date_len, util, [discount] * date_len, [model] * date_len]).T)
        res_util_df = pd.DataFrame(res_util, columns=['rent_day', 'res_day', 'util_rate', 'discount', 'res_model'])

        return res_util_df
