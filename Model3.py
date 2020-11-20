import os
import datetime as dt
from collections import defaultdict
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

class MODEL3(object):

    def __init__(self, curr_res_day: str):
        # Inintial dataset
        self.curr_res_day = curr_res_day
        self.curr_capa: dict = dict()
        self.season: pd.DataFrame = pd.DataFrame()
        self.dmd_pred: pd.DataFrame = pd.DataFrame()
        self.model_map: dict = dict()

        # dataset on prediction day
        self.exp_res: dict = dict()         # key: av / k3 / vl
        self.curr_res_cnt: dict = dict()    # key: av / k3 / vl
        self.curr_res_util: dict = dict()   # key: av / k3 / vl

        # Recommendation Input
        self.rec_input: dict = dict()
        self.exp_dmd_change: float = 0.

    def rec(self, pred_days: list, apply_day: str):
        self._load_init_data()

        summary = defaultdict(list)
        for pred_day in pred_days:
            # Load dataset
            self._load_data(pred_day=pred_day)

            # Preprocessing
            self._drop_column()                 # Drop columns
            self._rename_column()               # Rename columns
            self._merge_hx_curr_df()            # Set recommendation input
            self._fill_na(pred_day=pred_day)    # Forward fill
            self._set_exp_dmd_change(pred_day=pred_day)    #
            self._filter_date(apply_day=apply_day)
            self._scale_data()

            # Recommendation
            rec_output = self._rec_disc(pred_day=pred_day)
            rec_output_with_pol = self._apply_disc_policy(rec_output=rec_output)
            rec_output_rearranged = self._rearrange_column(rec_output=rec_output_with_pol)
            rec_output_renamed = self._rename_col_kor(rec_output=rec_output_rearranged)

            # Save result on each day
            self._save_result(pred_day=pred_day, rec_output=rec_output_renamed)

            fst_rec_data = self._get_fst_rec_data(rec_output=rec_output_renamed, pred_day=pred_day)

            for model, series in fst_rec_data.items():
                summary[model].append(series)

        # Filter and rearrange summary dataframe
        summary_df = self._filter_rearng_summary(summary=summary)

        # Save summary results
        self._save_summary_result(summary=summary_df, apply_day=apply_day)
        print("Recommendation Process is finished")

    def _save_result(self, pred_day: str, rec_output: dict):
        df_merged = pd.DataFrame()
        for model, df in rec_output.items():
            df_merged = pd.concat([df_merged, df], axis=1)

        save_path = os.path.join('..', 'result', 'data', 'recommend')
        df_merged.to_csv(os.path.join(save_path, 'original', 'rec(' + pred_day + ').csv'),
                         index=False, encoding='euc-kr')
        df_merged.T.to_csv(os.path.join(save_path, 'transpose', 'rec_T(' + pred_day + ').csv'),
                          header=False, encoding='euc-kr')

    def _save_result_BAK(self, pred_day: str, rec_output: dict):
        save_path = os.path.join('..', 'result', 'data', 'recommend')
        for model, df in rec_output.items():
            df.to_csv(os.path.join(save_path, 'original', 'rec_' + model + '(' + pred_day + ').csv'),
                      index=False, encoding='euc-kr')
            df.T.to_csv(os.path.join(save_path, 'transpose', 'rec_' + model + '_T' + '(' + pred_day + ').csv'),
                        header=False, encoding='euc-kr')

    @staticmethod
    def _rename_col_kor(rec_output: dict):
        rename_col = {
            'date': '날짜', 'lead_time': '리드타임', 'curr_cnt': '현재 예약건수', 'curr_util_time': '현재 가동대수(시간)',
            'curr_util_cnt': '현재 가동대수(일)', 'curr_util_rate': '현재 가동률', 'curr_disc': '현재 할인율',
            'avail_capa': '이용가능대수', 'exp_cnt': '기대 예약건수', 'exp_util_cnt': '기대 가동대수(시간)',
            'exp_util_rate': '기대가동률', 'exp_disc': '기대 할인율', 'rec_disc': '추천 할인율'}

        output_renamed = {}
        for model, df in rec_output.items():
            df = df.rename(columns=rename_col)
            df = df.reset_index(drop=True)
            output_renamed[model] = df

        return output_renamed

    @staticmethod
    def _rearrange_column(rec_output: dict):
        output_rearranged = {}
        reorder_cols = ['date', 'lead_time', 'curr_cnt', 'curr_util_time', 'curr_util_rate', 'curr_disc',
                        'avail_capa', 'exp_cnt', 'exp_util_cnt', 'exp_util_rate', 'exp_disc', 'rec_disc']
        for model, df in rec_output.items():
            df = df[reorder_cols]
            output_rearranged[model] = df

        return output_rearranged

    def _apply_disc_policy(self, rec_output: dict):
        rec_output_with_pol = {}
        vfunc = np.vectorize(self._conv_to_five_times)
        for model, df in rec_output.items():
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

    def _rec_disc(self, pred_day: str):
        rec_output = {}
        for model, df in self.rec_input.items():
            df['rec_disc_chg_rate'] = df[['curr_util_rate', 'exp_util_rate']].apply(self._get_rec, axis=1)
            df['rec_disc'] = df['curr_disc'] * (1 + df['rec_disc_chg_rate'] / 100)
            df = df.drop(columns=['rec_disc_chg_rate'], errors='ignore')
            rec_output[model] = df

        return rec_output

    def _scale_data(self):
        rec_input = {}
        for model, df in self.rec_input.items():
            df['curr_util_rate'] = df['curr_util_rate'] * 100
            df['exp_util_rate'] = df['exp_util_rate'] * 100
            df['exp_util_cnt'] = np.round(df['exp_util_cnt'].to_numpy(), 1)
            rec_input[model] = df
        self.rec_input = rec_input

    def _get_rec(self, x):
        return self._rec_disc_function(curr=x[0], exp=x[1], dmd=self.exp_dmd_change)

    def _merge_hx_curr_df(self):
        rec_input = {}
        for (model_hx, df_hx), (model_curr, df_curr) in zip(self.exp_res.items(), self.curr_res_cnt.items()):
            merged_model = pd.merge(df_hx, df_curr, how='left', on='lead_time',
                                    left_index=True, right_index=False)
            rec_input[model_hx] = merged_model

        for (model_hx, df_hx), (model_curr, df_curr) in zip(rec_input.items(), self.curr_res_util.items()):
            merged_model = pd.merge(df_hx, df_curr, how='left', on='lead_time',
                                    left_index=True, right_index=False)
            rec_input[model_hx] = merged_model
        self.rec_input = rec_input

    def _fill_na(self, pred_day: str):
        pred_mon = pred_day.split('-')[0] + pred_day.split('-')[1]
        for model, df in self.rec_input.items():
            df.fillna(method='ffill', axis=0, inplace=True)
            df['avail_capa'].fillna(self.curr_capa[(pred_mon, self.model_map[model])])
            df.fillna(0, axis=0, inplace=True)

    def _set_exp_dmd_change(self, pred_day: str):
        pred_datetime = dt.datetime(*list(map(int, pred_day.split('-'))))
        self.dmd_pred['date'] = pd.to_datetime(self.dmd_pred['date'], format='%Y%m')
        for date, dmd_chg in zip(self.dmd_pred['date'], self.dmd_pred['dmd_chg']):
            if (pred_datetime >= date) and (pred_datetime < date + relativedelta(months=1)):
                self.exp_dmd_change = dmd_chg

    def _filter_date(self, apply_day: str):

        for model, df in self.rec_input.items():
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            apply_datetime = dt.datetime(*list(map(int, apply_day.split('/'))))
            df = df[df['date'] >= apply_datetime]
            df['date'] = df['date'].apply(lambda x: dt.datetime.strftime(x, '%Y-%m-%d'))
            self.rec_input[model] = df

    def _load_init_data(self):
        # Capacity history of each car model
        load_path = os.path.join('..', 'input', 'capa', 'capa_curr.csv')
        curr_capa = pd.read_csv(load_path, delimiter='\t', dtype={'date': str, 'model': str, 'capa': int})
        self.curr_capa = {(month, model): capa for month, model, capa in zip(curr_capa['date'],
                                                                             curr_capa['model'],
                                                                             curr_capa['capa'])}

        # Seasonality
        load_path = os.path.join('..', 'input', 'seasonality')
        season_hx = pd.read_csv(os.path.join(load_path, 'seasonality_hx.csv'))
        season_curr = pd.read_csv(os.path.join(load_path, 'seasonality_curr.csv'), delimiter='\t')
        self.season = pd.concat([season_hx, season_curr])

        # Demand change prediction of jeju visitors
        load_path = os.path.join('..', 'result', 'data', 'model_1')
        self.dmd_pred = pd.read_csv(os.path.join(load_path, 'dmd_pred_2012_2102.csv'))

        self.model_map = {'av': 'AVANTE', 'k3': 'K3', 'vl': 'VELOSTER'}

    def _load_data(self, pred_day: str):
        load_path = os.path.join('..', 'result', 'data', 'prediction', 'original')
        self.exp_res = {'av': pd.read_csv(os.path.join(load_path, 'av', 'm2_pred(' + pred_day + ').csv')),
                        'k3': pd.read_csv(os.path.join(load_path, 'k3', 'm2_pred(' + pred_day + ').csv')),
                        'vl': pd.read_csv(os.path.join(load_path, 'vl', 'm2_pred(' + pred_day + ').csv'))}

        # Current reservation dataset
        load_path = os.path.join('..', 'result', 'data', 'reservation')
        res_cnt = pd.read_csv(os.path.join(load_path, 'res_curr_cnt(' + self.curr_res_day + ').csv'))
        res_util = pd.read_csv(os.path.join(load_path, 'res_curr_util(' + self.curr_res_day + ').csv'))

        self.curr_res_cnt = self._split_by_model(df=res_cnt[res_cnt['rent_day'] == pred_day])
        self.curr_res_util = self._split_by_model(df=res_util[res_util['rent_day'] == pred_day])

    def _rename_column(self):
        for df in self.curr_res_cnt.values():
            df.rename(columns={'res_cum_cnt': 'curr_cnt'}, inplace=True)

        for df in self.curr_res_util.values():
            df.rename(columns={'cum_util_time': 'curr_util_time',
                               'cum_util_cnt': 'curr_util_cnt',
                               'cum_util_time_rate': 'curr_util_rate'}, inplace=True)

        for df in self.exp_res.values():
            df.rename(columns={'exp_util': 'exp_util_rate'}, inplace=True)

    def _drop_column(self):
        # Drop columns
        for model_key, model_val in self.curr_res_cnt.items():
            model_val.drop(columns=['rent_day', 'model'], inplace=True, errors='ignore')

        for model_key, model_val in self.curr_res_util.items():
            model_val.drop(columns=['rent_day', 'model', 'cum_util_cnt_rate'], inplace=True, errors='ignore')

    def _split_by_model(self, df: pd.DataFrame):
        return {
            'av': df[df['model'] == 'AVANTE'],
            'k3': df[df['model'] == 'K3'],
            'vl': df[df['model'] == 'VELOSTER']}

    def _rec_disc_function(self, curr: float, exp: float, dmd: float):
        """
        Customized Exponential function
        curr_util: Current Utilization Rate
        d: Exp Demand Change Rate
        ex_util: Exp. Utilization rate
        """

        # Hyperparamters (need to tune)
        theta1 = 1  # ratio of increasing / decreasing magnitude : discount
        theta2 = 0.05  # ratio of increasing / decreasing magnitude : demand
        phi_low = 1.7  # 1 < phi_low < phi_high < 2
        phi_high = 1.2  # 1 < phi_low < phi_high < 2

        if dmd > 0:
            y = 1 - theta1 * (curr ** (phi_high ** (-1 * (curr * theta2 * dmd))) - exp)
        else:
            y = 1 - theta1 * (curr ** (phi_low ** (-1 * ((1 - curr) * theta2 * dmd))) - exp)

        return y

    def _get_fst_rec_data(self, rec_output: dict, pred_day: str):
        fst_rec_data = {}
        for model, df in rec_output.items():
            data = df.iloc[0, :]
            data['대여일'] = pred_day
            fst_rec_data[model] = data

        return fst_rec_data

    def _filter_rearng_summary(self, summary: dict):
        summary_df = {}
        for model, data in summary.items():
            df = pd.DataFrame(data)
            df = df.drop(columns=['날짜'])
            df = df[['대여일', '리드타임', '현재 예약건수', '현재 가동대수(시간)', '현재 가동률', '현재 할인율', '이용가능대수',
                     '기대 예약건수', '기대 가동대수(시간)', '기대가동률', '기대 할인율', '추천 할인율']]
            summary_df[model] = df

        return summary_df

    def _save_summary_result(self, summary: dict, apply_day: str):
        apply_day_str = ''.join(apply_day.split('/'))
        save_path = os.path.join('..', 'result', 'data', 'recommend', 'summary')
        for model, df in summary.items():
            df.to_csv(os.path.join(save_path, 'rec_summary_' + model + '(' + apply_day_str + ').csv'),
                      index=False, encoding='euc-kr')
            df.T.to_csv(os.path.join(save_path, 'rec_summary_' + model + '_T(' + apply_day_str + ').csv'),
                        header=False, encoding='euc-kr')