import os
import copy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# plt.style.use('fivethirtyeight')
from scipy.interpolate import interp1d

class Preprocessing(object):

    def __init__(self, root_dir: str, lead_times: list):
        self.root_dir = root_dir
        self.check_dir()
        self.lead_times: list = lead_times

    def check_dir(self):
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)

    def load_data(self, path: str, file_format: str, delimiter: str):
        """
        :param path: file path
        :param file_format: csv / excel
        :return: dataframe
        """
        file_path = os.path.join(self.root_dir, path)
        if file_format == 'csv':
            return pd.read_csv(file_path, delimiter=delimiter)
        elif file_format == 'excel':
            return pd.read_excel(file_path)

    @staticmethod
    def conv_to_datetime(df: pd.DataFrame, str_feat: str, datetime_feat: str, date_format='%Y%m%d'):
        df[datetime_feat] = pd.to_datetime(df[str_feat], format=date_format)
        return df

    @staticmethod
    def date_to_idx(df: pd.DataFrame, feature: str):
        df = df.set_index(feature)
        return df

    def count_discount_changes(self, df: pd.DataFrame, feature: str):
        """
        Count discount changes
        :param df: discount dataframe
        :param feature: count column
        :return: df
        """
        len_lt = len(self.lead_times)
        df_lt = copy.deepcopy(df[self.lead_times])
        df_lt = df_lt.fillna(value=0)

        cnt_feat = []
        for _, row in df_lt.iterrows():
            cnt = 0
            temp = row[0]
            for i in range(len_lt-1):
                if row[i+1] != temp:
                    cnt += 1
                    temp = row[i+1]
            cnt_feat.append(cnt)

        df[feature] = cnt_feat

        return df

    # Plot Method #
    @staticmethod
    def draw_histogram(df: pd.DataFrame, feature: str, title: str):
        plt.clf()
        ax = df.hist(column=feature, grid=False, bins=10,
                xlabelsize=8, ylabelsize=8)
        plt.title(title)
        plt.savefig(os.path.join('.', 'img', title + '.png'))

    @staticmethod
    def draw_line_plot(df: pd.DataFrame, x: str, y: str, interval: int, title: str):
        plt.clf()
        ax = df.plot.line(x=x, y=y,
                          figsize=(15, 6),
                          linewidth=0.8)
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))
        plt.gcf().autofmt_xdate()
        plt.title(title)
        plt.savefig(os.path.join('.', 'img', title + '.png'))
        # plt.show()

    def draw_plot_by_resample(self, df: pd.DataFrame, feat_date: str, feature: str, agg: str, title: str, rule='M'):
        plt.clf()
        df = self.date_to_idx(df=df, feature=feat_date)
        if agg == 'mean':
            df_sampled = df[feature].resample(rule=rule).mean()
        elif agg == 'sum':
            df_sampled = df[feature].resample(rule=rule).sum()

        ax = df_sampled.plot.line(x=df_sampled.index, y=df_sampled.values,
                                  figsize=(15, 6), linewidth=0.4)
        # ax.set_xticks([])
        # ax.set_xticks([], minor=True)
        plt.gcf().autofmt_xdate()
        plt.title(title)
        plt.savefig(os.path.join('.', 'img', title + '.png'))
        # plt.show()

    def draw_plot_with_hline(self, df: pd.DataFrame, line_feat: str, line_col: str, line_label: str,
                             h_line_val: float, hline_col: str, hline_label: str,
                             xlabel:str, ylabel: str, title: str):
        plt.clf()
        ax = df[line_feat].plot.line(figsize=(15, 6), linewidth=0.8, color=line_col,
                                     label=line_label)
        ax.hlines(y=h_line_val,
                  xmin=df.index[0],
                  xmax=df.index[-1],
                  ls='-', linewidth=0.8, color=hline_col,
                  label=hline_label)
        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        ax.legend()
        plt.title(title)
        plt.savefig(os.path.join('.', 'img', title + '.png'))
        # plt.show()