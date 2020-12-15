from Utility import Utility

import os
import datetime as dt
from collections import defaultdict
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


class DiscRecommend(object):

    def __init__(self, res_status_ud_day: str, apply_day: str, model_detail: str,
                 disc_confirm_last_week: str):
        self.utility = Utility
        # Initial data
        self.res_update_day = res_status_ud_day
        self.apply_day = apply_day
        self.disc_confirm_last_week = disc_confirm_last_week
        self.type_apply = model_detail
        self.capa_re: dict = dict()
        self.season: pd.DataFrame = pd.DataFrame()
        self.dmd_pred: pd.DataFrame = pd.DataFrame()

        # Data types
        self.type_data: list = self.utility.TYPE_DATA
        self.type_data_map: dict = self.utility.TYPE_DATA_MAP

        # Car types
        self.model_nm_map: dict = self.utility.MODEL_NAME_MAP
        self.model_nm_map_rev: dict = self.utility.MODEL_NAME_MAP_REV
        self.type_group: list = self.utility.TYPE_GROUP
        self.type_model: list = self.utility.TYPE_MODEL
        self.type_apply: dict = {'model': self.type_group, 'car': self.type_model}

        # dataset on prediction day
        self.res_pred: dict = dict()       # key: av / k3 / vl / su
        self.res_cnt_re: dict = dict()    # key: av / k3 / vl / su
        self.res_util_re: dict = dict()   # key: av / k3 / vl / su
        self.disc_confirm: dict = {}

        # Recommendation Input
        self.rec_input: dict = dict()
        self.exp_dmd_change: float = 0.

    def rec(self, pred_days: list, type_apply: str):
        self._set_necessary_data(type_apply=type_apply)
        self.disc_confirm = self._get_disc_confirm_last_week()

        summary = defaultdict(list)
        for pred_day in pred_days:
            # Load dataset
            self._load_data(pred_day=pred_day)

            # Preprocessing
            self._drop_column()                 # Drop columns
            self._rename_column()               # Rename columns
            rec_input = self._merge_hx_curr_df()    # Set recommendation input
            rec_input = self._add_feature(input_dict=rec_input, pred_day=pred_day)
            rec_input = self._fill_na(input_dict=rec_input)    # Forward fill
            self._set_exp_dmd_change(pred_day=pred_day)    #
            rec_input = self._filter_date(input_dict=rec_input)
            rec_input = self._scale_data(input_dict=rec_input)

            # Recommendation
            output = self._rec_disc(input_dict=rec_input)

            # Apply discount policy
            output_with_pol = self._apply_disc_policy(output=output)
            # Change data scale
            output_rescaled = self._chg_data_scale(output=output_with_pol)
            # Rearrange dataset columns
            output_rearranged = self._rearrange_column(output=output_rescaled)
            # Rename columns
            output_renamed = self._rename_col_kor(output=output_rearranged)

            # Save result on each day
            self._save_result(output=output_renamed, pred_day=pred_day)
            # Get recommendation data of latest day
            fst_rec_data = self._get_fst_rec_data(output=output_renamed, pred_day=pred_day)

            for model, series in fst_rec_data.items():
                summary[model].append(series)

        # Filter and rearrange summary dataframe
        summary_df = self._filter_arrange_summary(summary=summary)

        # Save summary results
        self._save_summary_result(summary=summary_df)
        print("Recommendation Process is finished")

    def _get_disc_confirm_last_week(self):
        # Initial capacity of each model
        load_path = os.path.join('..', 'input', 'disc_confirm')
        disc_comfirm = pd.read_csv(os.path.join(load_path, ''.join(['disc_confirm_', self.disc_confirm_last_week,
                                                '.csv'])), delimiter='\t', dtype={'date': str, 'disc': int})
        disc_comfirm['date'] = pd.to_datetime(disc_comfirm['date'], format='%Y%m%d')
        disc_comfirm_dict = defaultdict(dict)
        for model, date, disc in zip(disc_comfirm['model'], disc_comfirm['date'], disc_comfirm['disc']):
            disc_comfirm_dict[self.model_nm_map[model]].update({date: disc})

        return disc_comfirm_dict

    def _set_necessary_data(self, type_apply: str):
        # Capacity history of each car model
        capa_re = self.utility.get_capacity(time='re', type_apply=type_apply)
        capa_re_unavail = self.utility.get_capacity(time='re', type_apply=type_apply, unavail=True)
        capa_re_unavail = capa_re_unavail.rename(columns={'capa': 'unavail'})
        capa_re = self._conv_mon_to_day(df=capa_re, end_day='28')
        capa_re = self._apply_unavail_capa(capa=capa_re, capa_unavail=capa_re_unavail)

        self.capa_re = {(date, model): capa for date, model, capa in zip(capa_re['date'],
                                                                         capa_re['model'],
                                                                         capa_re['capa'])}

        # Seasonality
        season_hx = self.utility.get_season(time='hx')
        season_re = self.utility.get_season(time='re')
        self.season = pd.concat([season_hx, season_re])

        # Demand change prediction of jeju visitors
        load_path = os.path.join('..', 'result', 'data', 'model_1')
        self.dmd_pred = pd.read_csv(os.path.join(load_path, 'dmd_pred_2012_2102.csv'))

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

    def _load_data(self, pred_day: str):
        load_path = os.path.join('..', 'result', 'data', 'prediction', self.apply_day)
        detail_type = self.type_apply[self.type_apply]
        res_exp = {}
        for model in detail_type:
            res_exp[model] = pd.read_csv(os.path.join(load_path, model, 'res_pred(' + pred_day + ').csv'))
        self.res_pred = res_exp

        # Current reservation dataset
        load_path = os.path.join('..', 'result', 'data', 'model_2', self.res_update_day, self.type_apply)
        res_re = defaultdict(dict)
        for data_type in ['cnt_cum', 'util_cum']:
            for model in detail_type:
                df = pd.read_csv(os.path.join(load_path, data_type, data_type + '_' + model + '.csv'))
                df = df[df['rent_day'] == pred_day]
                res_re[data_type].update({model: df})

        self.res_cnt_re = res_re['cnt_cum']
        self.res_util_re = res_re['util_cum']

    def _drop_column(self):
        # Drop columns
        for model_key, model_val in self.res_cnt_re.items():
            model_val.drop(columns=['rent_day', 'res_day', 'res_model', 'lead_time_vec', 'seasonality'],
                           inplace=True, errors='ignore')

        for model_key, model_val in self.res_util_re.items():
            model_val.drop(columns=['rent_day', 'res_day', 'res_model', 'cum_util_cnt_rate', 'disc_mean',
                                    'lead_time_vec', 'seasonality'],
                           inplace=True, errors='ignore')

    def _rename_column(self):
        for df in self.res_cnt_re.values():
            df.rename(columns={'cnt_cum': 'curr_cnt',
                               'disc_mean': 'curr_disc_apply'}, inplace=True)

        for df in self.res_util_re.values():
            df.rename(columns={'util_cum': 'curr_util_time',
                               'util_rate_cum': 'curr_util_rate'}, inplace=True)

        for df in self.res_pred.values():
            df.rename(columns={'exp_util': 'exp_util_rate'}, inplace=True)

    def _merge_hx_curr_df(self):
        rec_input = {}
        for (model_hx, df_hx), (model_curr, df_curr) in zip(self.res_pred.items(), self.res_cnt_re.items()):
            merged_model = pd.merge(df_hx, df_curr, how='left', on=['lead_time'],
                                    left_index=True, right_index=False)
            rec_input[model_hx] = merged_model

        for (model_hx, df_hx), (model_curr, df_curr) in zip(rec_input.items(), self.res_util_re.items()):
            merged_model = pd.merge(df_hx, df_curr, how='left', on=['lead_time'],
                                    left_index=True, right_index=False)
            rec_input[model_hx] = merged_model

        return rec_input

    def _add_feature(self, input_dict: dict, pred_day: str):
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        for model, df in input_dict.items():
            df['avail_capa'] = self.capa_re[(pred_datetime, self.model_nm_map_rev[model])] - df['curr_util_time']
            df['applied_disc'] = self.disc_confirm[model][pred_datetime]

        return input_dict

    @staticmethod
    def _fill_na(input_dict: dict):
        for model, df in input_dict.items():
            df.fillna(method='ffill', axis=0, inplace=True)
            df.fillna(0, axis=0, inplace=True)

        return input_dict

    def _set_exp_dmd_change(self, pred_day: str):
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        self.dmd_pred['date'] = pd.to_datetime(self.dmd_pred['date'], format='%Y%m')
        for date, dmd_chg in zip(self.dmd_pred['date'], self.dmd_pred['dmd_chg']):
            if (pred_datetime >= date) and (pred_datetime < date + relativedelta(months=1)):
                self.exp_dmd_change = dmd_chg

    def _filter_date(self, input_dict: dict):
        for model, df in input_dict.items():
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            apply_datetime = dt.datetime.strptime(self.apply_day, '%Y%m%d')
            df = df[df['date'] >= apply_datetime]
            df['date'] = df['date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
            input_dict[model] = df

        return input_dict

    @staticmethod
    def _scale_data(input_dict: dict):
        rec_input = {}
        for model, df in input_dict.items():
            df['curr_util_rate'] = df['curr_util_rate'] * 100
            df['exp_util_rate'] = df['exp_util_rate'] * 100
            df['exp_util_cnt'] = np.round(df['exp_util_cnt'].to_numpy(), 1)
            rec_input[model] = df

        return input_dict

    def _rec_disc(self, input_dict: dict):
        rec_output = {}
        for model, df in input_dict.items():
            # util_rate: float (0.xx)
            df['rec_disc_chg_rate'] = df[['curr_util_rate', 'exp_util_rate']].apply(self._get_rec, axis=1)
            df['rec_disc'] = df['applied_disc'] * (1 + df['rec_disc_chg_rate'] / 100)
            df = df.drop(columns=['rec_disc_chg_rate'], errors='ignore')
            rec_output[model] = df

        return rec_output

    def _get_rec(self, x):
        return self._rec_disc_function(curr=x[0], exp=x[1], dmd=0)
        # return self._rec_disc_function(curr=x[0], exp=x[1], dmd=self.exp_dmd_change)

    @staticmethod
    def _rec_disc_function(curr: float, exp: float, dmd=0):
        """
        Recommendation Objective function
        curr: Current Utilization Rate (0 < curr < 1)
        exp: Expected Utilization rate (0 < exp < 1)
        dmd: Expected Demand Change Rate (float)
        """
        # Hyper-parameters (need to tune)
        theta_1 = 1     # Ratio of Decreasing magnitude
        if curr < exp:  # Ratio of increasing magnitude
            theta_1 = 0.7
        theta_2 = 1     # Ratio of Demand magnitude

        # Suggestion curve bend
        # 1 < phi_low < phi_high < 2
        # phi_low = 1.7
        phi_high = 1.2

        y = 1 - theta_1 * (curr ** (phi_high ** (-1 * curr)) - exp * (1 - theta_2 * dmd))

        return y

    @staticmethod
    def _rec_disc_function_bak(curr: float, exp: float, dmd: float):
        """
        Recommendation Objective function
        curr: Current Utilization Rate
        exp: Expected Utilization rate
        dmd: Exp Demand Change Rate (Gaussian scale)
        """
        # Hyper-parameters (need to tune)
        theta1 = 1          # Ratio of Decreasing magnitude
        if curr < exp:      # Ratio of increasing magnitude
            theta1 = 0.5
        theta2 = 0.05  # ratio of demand increasing / decreasing magnitude
        # Suggestion curve bend
        # 1 < phi_low < phi_high < 2
        phi_low = 1.7
        phi_high = 1.2

        if dmd > 0:
            y = 1 - theta1 * (curr ** (phi_high ** (-1 * (curr * theta2 * dmd))) - exp)
        else:
            y = 1 - theta1 * (curr ** (phi_low ** (-1 * ((1 - curr) * theta2 * dmd))) - exp)

        return y

    def _apply_disc_policy(self, output: dict):
        rec_output_with_pol = {}
        vfunc = np.vectorize(self._conv_to_five_times)
        for model, df in output.items():
            # Set maximum discount rate: 80%
            df['rec_disc'] = np.where(df['rec_disc'] > 80, 80, df['rec_disc'])
            # Convert discount to five times values
            df['rec_disc'] = vfunc(df['rec_disc'].to_numpy())

            rec_output_with_pol[model] = df

        return rec_output_with_pol

    @staticmethod
    def _conv_to_five_times(discount):
        if discount % 5 >= 2.5:
            return (discount // 5 + 1) * 5
        else:
            return (discount // 5) * 5

    @staticmethod
    def _chg_data_scale(output: dict):
        for model, df in output.items():
            df['curr_util_time'] = np.round(df['curr_util_time'].to_numpy(), 1)
            df['curr_util_rate'] = np.round(df['curr_util_rate'].to_numpy(), 1)
            df['curr_disc_apply'] = np.round(df['curr_disc_apply'].to_numpy(), 1)
            df['avail_capa'] = np.round(df['avail_capa'].to_numpy(), 1)

        return output

    @staticmethod
    def _rearrange_column(output: dict):
        output_rearranged = {}
        reorder_cols = ['date', 'lead_time', 'curr_cnt', 'curr_util_time', 'curr_util_rate', 'curr_disc',
                        'curr_disc_apply', 'avail_capa', 'exp_cnt', 'exp_util_cnt', 'exp_util_rate',
                        'exp_disc', 'rec_disc']
        for model, df in output.items():
            df = df[reorder_cols]
            output_rearranged[model] = df

        return output_rearranged

    @staticmethod
    def _rename_col_kor(output: dict):
        rename_col = {
            'date': '날짜', 'lead_time': '리드타임', 'curr_cnt': '현재 예약건수', 'curr_util_time': '현재 가동대수(시간)',
            'curr_util_cnt': '현재 가동대수(일)', 'curr_util_rate': '현재 가동률', 'curr_disc': '현재 할인율',
            'curr_disc_apply': '현재 결제할인율(평균)', 'avail_capa': '이용가능대수', 'exp_cnt': '기대 예약건수',
            'exp_util_cnt': '기대 가동대수(시간)', 'exp_util_rate': '기대가동률', 'exp_disc': '기대 할인율',
            'rec_disc': '추천 할인율'}

        output_renamed = {}
        for model, df in output.items():
            df = df.rename(columns=rename_col)
            df = df.reset_index(drop=True)
            output_renamed[model] = df

        return output_renamed

    def _save_result(self, output: dict, pred_day: str):
        df_merged = pd.DataFrame()
        for model, df in output.items():
            df_merged = pd.concat([df_merged, df], axis=1)

        save_path = os.path.join('..', 'result', 'data', 'recommend', 'lead_time', self.apply_day)

        # Make directory
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(os.path.join(save_path, 'original'))
            os.mkdir(os.path.join(save_path, 'transpose'))

        # Save results
        df_merged.to_csv(os.path.join(save_path, 'original', 'rec(' + pred_day + ').csv'),
                         index=False, encoding='euc-kr')
        df_merged.T.to_csv(os.path.join(save_path, 'transpose', 'rec_T(' + pred_day + ').csv'),
                           header=False, encoding='euc-kr')

    @staticmethod
    def _get_fst_rec_data(output: dict, pred_day: str):
        fst_rec_data = {}
        for model, df in output.items():
            data = df.iloc[0, :]
            data['대여일'] = pred_day
            fst_rec_data[model] = data

        return fst_rec_data

    @staticmethod
    def _filter_arrange_summary(summary: dict):
        summary_df = {}
        for model, data in summary.items():
            df = pd.DataFrame(data)
            df = df.drop(columns=['날짜'])
            df = df[['대여일', '리드타임', '현재 예약건수', '현재 가동대수(시간)', '현재 가동률', '현재 할인율',
                     '현재 결제할인율(평균)', '이용가능대수', '기대 예약건수', '기대 가동대수(시간)', '기대가동률',
                     '기대 할인율', '추천 할인율']]
            summary_df[model] = df

        return summary_df

    def _save_summary_result(self, summary: dict):
        save_path = os.path.join('..', 'result', 'data', 'recommend', 'summary', self.apply_day)

        # Make directory
        if not os.path.exists(save_path):
            os.mkdir(os.path.join(save_path))
            os.mkdir(os.path.join(save_path, 'original'))
            os.mkdir(os.path.join(save_path, 'transpose'))

        # Save results
        for model, df in summary.items():
            df.to_csv(os.path.join(save_path, 'original', 'rec_summary_' + model + '.csv'),
                      index=False, encoding='euc-kr')
            df.T.to_csv(os.path.join(save_path, 'transpose', 'rec_summary_' + model + '_T.csv'),
                        header=False, encoding='euc-kr')
