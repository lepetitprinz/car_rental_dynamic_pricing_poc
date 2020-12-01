import os
import datetime as dt
from collections import defaultdict
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd


class DiscRecommend(object):

    def __init__(self, res_update_day: str):
        # Inintial dataset
        self.res_update_day = res_update_day
        self.capa_re: dict = dict()
        self.season: pd.DataFrame = pd.DataFrame()
        self.dmd_pred: pd.DataFrame = pd.DataFrame()
        self.model_type: list = ['av', 'k3', 'vl', 'su']
        self.model_map = {'av': 'AVANTE', 'k3': 'K3', 'vl': 'VELOSTER', 'su': 'SOUL'}

        # dataset on prediction day
        self.res_exp: dict = dict()       # key: av / k3 / vl / su
        self.res_cnt_re: dict = dict()    # key: av / k3 / vl / su
        self.res_util_re: dict = dict()   # key: av / k3 / vl / su

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
            rec_input = self._merge_hx_curr_df()    # Set recommendation input
            rec_input = self._add_feature(input_dict=rec_input, pred_day=pred_day)
            rec_input = self._fill_na(input_dict=rec_input)    # Forward fill
            self._set_exp_dmd_change(pred_day=pred_day)    #
            rec_input = self._filter_date(input_dict=rec_input, apply_day=apply_day)
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
            self._save_result(output=output_renamed, pred_day=pred_day, apply_day=apply_day)
            # Get recommendation data of latest day
            fst_rec_data = self._get_fst_rec_data(output=output_renamed, pred_day=pred_day)

            for model, series in fst_rec_data.items():
                summary[model].append(series)

        # Filter and rearrange summary dataframe
        summary_df = self._filter_arrange_summary(summary=summary)

        # Save summary results
        self._save_summary_result(summary=summary_df, apply_day=apply_day)
        print("Recommendation Process is finished")

    def _load_init_data(self):
        # Capacity history of each car model
        load_path = os.path.join('..', 'input', 'capa', 'capa_curr_model.csv')
        capa_re = pd.read_csv(load_path, delimiter='\t', dtype={'date': str, 'model': str, 'capa': int})
        self.capa_re = {(month, model): capa for month, model, capa in zip(capa_re['date'],
                                                                           capa_re['model'],
                                                                           capa_re['capa'])}

        # Seasonality
        load_path = os.path.join('..', 'input', 'seasonality')
        season_hx = pd.read_csv(os.path.join(load_path, 'seasonality_hx.csv'))
        season_re = pd.read_csv(os.path.join(load_path, 'seasonality_re.csv'), delimiter='\t')
        self.season = pd.concat([season_hx, season_re])

        # Demand change prediction of jeju visitors
        load_path = os.path.join('..', 'result', 'data', 'model_1')
        self.dmd_pred = pd.read_csv(os.path.join(load_path, 'dmd_pred_2012_2102.csv'))

    def _load_data(self, pred_day: str):
        load_path = os.path.join('..', 'result', 'data', 'prediction')
        self.res_exp = {'av': pd.read_csv(os.path.join(load_path, 'av', 'm2_pred(' + pred_day + ').csv')),
                        'k3': pd.read_csv(os.path.join(load_path, 'k3', 'm2_pred(' + pred_day + ').csv')),
                        'vl': pd.read_csv(os.path.join(load_path, 'vl', 'm2_pred(' + pred_day + ').csv')),
                        'su': pd.read_csv(os.path.join(load_path, 'su', 'm2_pred(' + pred_day + ').csv'))}

        # Current reservation dataset
        load_path = os.path.join('..', 'result', 'data', 'model_2', 're', 'model')
        res_re = defaultdict(dict)
        for data_type in ['res', 'util']:
            for model in self.model_type:
                df = pd.read_csv(os.path.join(load_path, 'disc_' + data_type + '_cum_' + model + '.csv'))
                df = df[df['rent_day'] == pred_day]
                res_re[data_type].update({model: df})

        self.res_cnt_re = res_re['res']
        self.res_util_re = res_re['util']

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

        for df in self.res_exp.values():
            df.rename(columns={'exp_util': 'exp_util_rate'}, inplace=True)

    @staticmethod
    def _chg_data_scale(output: dict):
        for model, df in output.items():
            df['curr_util_time'] = np.round(df['curr_util_time'].to_numpy(), 1)
            df['curr_util_rate'] = np.round(df['curr_util_rate'].to_numpy(), 1)
            df['curr_disc_apply'] = np.round(df['curr_disc_apply'].to_numpy(), 1)
            df['avail_capa'] = np.round(df['avail_capa'].to_numpy(), 1)

        return output

    @staticmethod
    def _save_result(output: dict, pred_day: str, apply_day: str):
        df_merged = pd.DataFrame()
        for model, df in output.items():
            df_merged = pd.concat([df_merged, df], axis=1)

        apply_day_str = ''.join(apply_day.split('/'))
        save_path = os.path.join('..', 'result', 'data', 'recommend', 'lead_time', apply_day_str)

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

    def _merge_hx_curr_df(self):
        rec_input = {}
        for (model_hx, df_hx), (model_curr, df_curr) in zip(self.res_exp.items(), self.res_cnt_re.items()):
            merged_model = pd.merge(df_hx, df_curr, how='left', on=['lead_time'],
                                    left_index=True, right_index=False)
            rec_input[model_hx] = merged_model

        for (model_hx, df_hx), (model_curr, df_curr) in zip(rec_input.items(), self.res_util_re.items()):
            merged_model = pd.merge(df_hx, df_curr, how='left', on=['lead_time'],
                                    left_index=True, right_index=False)
            rec_input[model_hx] = merged_model

        return rec_input

    def _add_feature(self, input_dict: dict, pred_day: str):
        pred_mon = pred_day.split('-')[0] + pred_day.split('-')[1]
        for model, df in input_dict.items():
            df['avail_capa'] = self.capa_re[(pred_mon, self.model_map[model])] - df['curr_util_time']

        return input_dict

    @staticmethod
    def _conv_to_five_times(discount):
        if discount % 5 >= 2.5:
            return (discount // 5 + 1) * 5
        else:
            return (discount // 5) * 5

    def _rec_disc(self, input_dict: dict):
        rec_output = {}
        for model, df in input_dict.items():
            df['rec_disc_chg_rate'] = df[['curr_util_rate', 'exp_util_rate']].apply(self._get_rec, axis=1)
            df['rec_disc'] = df['curr_disc'] * (1 + df['rec_disc_chg_rate'] / 100)
            df = df.drop(columns=['rec_disc_chg_rate'], errors='ignore')
            rec_output[model] = df

        return rec_output

    def _get_rec(self, x):
        return self._rec_disc_function(curr=x[0], exp=x[1], dmd=0)
        # return self._rec_disc_function(curr=x[0], exp=x[1], dmd=self.exp_dmd_change)

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

    @staticmethod
    def _filter_date(input_dict: dict,  apply_day: str):
        for model, df in input_dict.items():
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            apply_datetime = dt.datetime(*list(map(int, apply_day.split('/'))))
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

    @staticmethod
    def _rec_disc_function(curr: float, exp: float, dmd: float):
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

    @staticmethod
    def _save_summary_result(summary: dict, apply_day: str):
        apply_day_str = ''.join(apply_day.split('/'))
        save_path = os.path.join('..', 'result', 'data', 'recommend', 'summary', apply_day_str)

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
