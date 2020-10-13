# from DataHandler import DataHandler
from Preprocessing import Preprocessing

import os
import copy
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Preprocessing Setting
root_dir = os.path.join('..', 'input')
lead_times = ['M3', 'M2', 'M1', 'W4', 'W3', 'W2', 'W1']

# Initialize Preprocessing class
prep = Preprocessing(root_dir=root_dir,
                     lead_times=lead_times)

# Discount dataset
# Load data: discount
discount = prep.load_data(path='discount_by_lead_time.csv', file_format='csv', delimiter='\t')

#
discount = prep.conv_to_datetime(df=discount, str_feat='YYYYMMDD', datetime_feat='date')

# Make discount change feature
discount = prep.count_discount_changes(df=discount, feature='chng_cnt')

discount_17 = discount[discount['YYYYMMDD'] <= 20171231]
discount_18 = discount[(discount['YYYYMMDD'] > 20171231) & (discount['YYYYMMDD'] <= 20181231)]
discount_19 = discount[(discount['YYYYMMDD'] > 20181231) & (discount['YYYYMMDD'] <= 20191231)]
discount_20 = discount[discount['YYYYMMDD'] > 20191231]

# View histogram: count of discount change
prep.draw_histogram(df=discount, feature='chng_cnt', title='Counts of discount change histogram-total')
prep.draw_histogram(df=discount_17, feature='chng_cnt', title='Counts of discount change histogram-2017')
prep.draw_histogram(df=discount_18, feature='chng_cnt', title='Counts of discount change histogram-2018')
prep.draw_histogram(df=discount_19, feature='chng_cnt', title='Counts of discount change histogram-2019')


# View line plot: count of discount change
prep.draw_line_plot(df=discount_17, x='date', y='chng_cnt', interval=1, title='2017 Discount Change')
prep.draw_line_plot(df=discount_18, x='date', y='chng_cnt', interval=1, title='2018 Discount Change')
prep.draw_line_plot(df=discount_19, x='date', y='chng_cnt', interval=1, title='2019 Discount Change')
prep.draw_line_plot(df=discount_20, x='date', y='chng_cnt', interval=1, title='2020 Discount Change')

# View monthly average count plot
prep.draw_plot_by_resample(df=discount, feat_date='date', feature='chng_cnt', agg='mean',
                           title='Discount Change Plot')

# Reservation dataset
res_17 = prep.load_data(path='reservation_17.csv', file_format='csv', delimiter='\t')
res_18 = prep.load_data(path='reservation_18.csv', file_format='csv', delimiter='\t')
res_19 = prep.load_data(path='reservation_19.csv', file_format='csv', delimiter='\t')

# Merge 17 ~ 19 years reservation dataset
res = pd.concat([res_17, res_18, res_19], axis=0, ignore_index=True)

# Renames columns
col_remap = {'예약경로': 'rev_route', '예약경로명': 'rev_route_nm', '계약번호': 'rev_num',
             '고객': 'cust_num', '고객구분': 'cust_kind', '고객구분명': 'cust_kind_nm',
             '총 청구액(VAT포함)': 'tot_bill', '총 수납금액(VAT포함)': 'tot_pay',
             '총 잔액(VAT포함)': 'tot_balance', '예약모델': 'rev_model', '예약모델명': 'rev_model_nm',
             '차급': 'car_grd', '대여일': 'rent_day', '대여시간': 'rent_time',
             '반납일': 'return_day', '반납시간': 'return_time', '대여기간(일)': 'rent_period',
             '대여기간(시간)': 'rent_period_time', '실반납일시': 'real_return_time',
             '실대여기간(일)': 'real_rent_period', '실대여기간(시간)': 'real_rent_time', '차량대여요금(VAT포함)': 'car_rent_fee',
             'CDW가입여부': 'cdw_yn', 'CDW요금구분': 'cdw_fee_kind', 'CDW요금구분명': 'cdw_fee_kind_nm',
             'CDW요금': 'cdw_fee', '회원등급': 'member_grd', '차종': 'car_kind', '구매목적': 'rent_purpose',
             '내부매출액': 'in_sales', '수납': 'pay_kind', '예약일자': 'rev_date', '할인유형': 'discount_kind',
             '할인유형명': 'discount_kind_nm', '적용할인명': 'apply_discount_nm'}

res.rename(columns=col_remap, inplace=True)

# Drop columns
drop_cols = ['tot_pay', 'tot_balance', 'in_sales',
             'cdw_yn', 'cdw_fee_kind', 'cdw_fee_kind_nm',
             'car_grd', 'car_kind', 'rent_purpose', 'pay_kind']
res = res.drop(columns=drop_cols, axis=1)

res = prep.conv_to_datetime(df=res, str_feat='rent_day', datetime_feat='rent_day', date_format='%Y-%m-%d')
res = prep.conv_to_datetime(df=res, str_feat='return_day', datetime_feat='return_day', date_format='%Y-%m-%d')
res = prep.conv_to_datetime(df=res, str_feat='rev_date', datetime_feat='rev_date', date_format='%Y-%m-%d')

# Calculate Sales
res['car_rent_fee'] = res['car_rent_fee'].astype(int)
res['cdw_fee'] = res['cdw_fee'].astype(int)
res['sales'] = res['car_rent_fee'] + res['cdw_fee']

# Calculate customer reservation lead time
res['lead_time'] = res['rent_day'] - res['rev_date']
res['lead_time'] = res['lead_time'].apply(lambda x: x.days)
res = res.sort_values(by=['rent_day'])

# Calculate sales per times
res['rent_tot_times'] = res['rent_period'] * 24 + res['rent_period_time']
res['sales_per_times'] = res['sales'] / res['rent_tot_times']


# Separate dataset for each year
res_17 = copy.deepcopy(res[(res['rent_day'] > pd.to_datetime('20161231', format='%Y-%m-%d')) &
                           (res['rent_day'] <= pd.to_datetime('20171231', format='%Y-%m-%d'))])
res_18 = copy.deepcopy(res[(res['rent_day'] > pd.to_datetime('20171231', format='%Y-%m-%d')) &
                           (res['rent_day'] <= pd.to_datetime('20181231', format='%Y-%m-%d'))])
res_19 = copy.deepcopy(res[(res['rent_day'] > pd.to_datetime('20181231', format='%Y-%m-%d')) &
                           (res['rent_day'] <= pd.to_datetime('20191231', format='%Y-%m-%d'))])

# ----------------------------- #
# Analysis based on rental day
# ----------------------------- #
res_17_grp_by_rent_day = res_17.groupby('rent_day').mean()
res_18_grp_by_rent_day = res_18.groupby('rent_day').mean()
res_19_grp_by_rent_day = res_19.groupby('rent_day').mean()

# Mean of lead time each year
res_17_lead_time_mean = res_17['lead_time'].mean()
res_18_lead_time_mean = res_18['lead_time'].mean()
res_19_lead_time_mean = res_19['lead_time'].mean()

# 2.1 Lead Time Trend

# Annotation List
# 2017
lt_annot_2017 = [{'text': 'Holidays', 'xy': (dt.datetime(2017, 1, 27), 21), 'xytext': (dt.datetime(2017, 2, 4), 27),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Peak season', 'xy': (dt.datetime(2017, 5, 1), 40), 'xytext': (dt.datetime(2017, 5, 10), 46),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Peak season', 'xy': (dt.datetime(2017, 7, 26), 42), 'xytext': (dt.datetime(2017, 6, 12), 35),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2017, 9, 26), 50), 'xytext': (dt.datetime(2017, 8, 24), 43),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)}]
# 2018
lt_annot_2018 = [{'text': 'Holidays', 'xy': (dt.datetime(2018, 2, 16), 26), 'xytext': (dt.datetime(2018, 2, 26), 31),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2018, 5, 6), 27), 'xytext': (dt.datetime(2018, 5, 16), 32),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Peak season', 'xy': (dt.datetime(2018, 7, 26), 40), 'xytext': (dt.datetime(2018, 6, 12), 35),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2018, 9, 22), 31), 'xytext': (dt.datetime(2018, 8, 22), 36),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)}]
# 2019
lt_annot_2019 = [{'text': 'Holidays', 'xy': (dt.datetime(2019, 2, 4), 19), 'xytext': (dt.datetime(2019, 1, 5), 24),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2019, 3, 1), 26), 'xytext': (dt.datetime(2019, 1, 30), 31),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2019, 5, 4), 25), 'xytext': (dt.datetime(2019, 4, 4), 30),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2019, 6, 6), 32), 'xytext': (dt.datetime(2019, 5, 6), 37),
                  'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Peak season', 'xy': (dt.datetime(2019, 7, 26), 41), 'xytext': (dt.datetime(2019, 6, 12), 36),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2019, 9, 13), 34), 'xytext': (dt.datetime(2019, 9, 23), 29),
                 'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                 {'text': 'Holidays', 'xy': (dt.datetime(2019, 12, 24), 34), 'xytext': (dt.datetime(2019, 11, 20), 29),
                  'arrowprops': dict(facecolor='darkblue', shrink=0.05)}]

prep.draw_plot_with_hline(df=res_17_grp_by_rent_day,
                          line_feat='lead_time', line_col='k', line_label='Lead Time',
                          h_line_val=res_17_lead_time_mean, hline_col='firebrick', hline_label='Average Lead Time',
                          xlabel='Rental Day', ylabel='Lead Time (days)', title='Lead Time (2017)', annot=lt_annot_2017)

prep.draw_plot_with_hline(df=res_18_grp_by_rent_day,
                          line_feat='lead_time', line_col='k', line_label='Lead Time',
                          h_line_val=res_18_lead_time_mean, hline_col='firebrick', hline_label='Average Lead Time',
                          xlabel='Rental Day', ylabel='Lead Time (days)', title='Lead Time (2018)', annot=lt_annot_2018)

prep.draw_plot_with_hline(df=res_19_grp_by_rent_day,
                          line_feat='lead_time', line_col='k', line_label='Lead Time',
                          h_line_val=res_19_lead_time_mean, hline_col='firebrick', hline_label='Average Lead Time',
                          xlabel='Rental Day', ylabel='Lead Time (days)', title='Lead Time (2019)', annot=lt_annot_2019)

# 2.2 Sales by Rental Days
res_17_sales_mean = res_17['sales'].mean()
res_18_sales_mean = res_18['sales'].mean()
res_19_sales_mean = res_19['sales'].mean()

# 2.2.1 Sales Trend

# Annotation List
# 2017
sales_annot_2017 = [{'text': 'Holidays', 'xy': (dt.datetime(2017, 1, 27), 175000),
                     'xytext': (dt.datetime(2017, 2, 4), 195000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Peak season', 'xy': (dt.datetime(2017, 5, 1), 230000),
                     'xytext': (dt.datetime(2017, 5, 10), 250000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Peak season', 'xy': (dt.datetime(2017, 7, 26), 330000),
                     'xytext': (dt.datetime(2017, 6, 12), 310000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2017, 9, 26), 300000),
                     'xytext': (dt.datetime(2017, 8, 24), 280000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)}]
# 2018
sales_annot_2018 = [{'text': 'Holidays', 'xy': (dt.datetime(2018, 2, 16), 210000),
                     'xytext': (dt.datetime(2018, 2, 26), 240000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2018, 5, 6), 200000),
                     'xytext': (dt.datetime(2018, 5, 16), 230000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Peak season', 'xy': (dt.datetime(2018, 7, 26), 320000),
                     'xytext': (dt.datetime(2018, 6, 12), 290000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2018, 9, 22), 270000),
                     'xytext': (dt.datetime(2018, 8, 22), 300000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)}]
# 2019
sales_annot_2019 = [{'text': 'Holidays', 'xy': (dt.datetime(2019, 2, 4), 170000),
                     'xytext': (dt.datetime(2019, 1, 5), 200000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2019, 3, 1), 135000),
                     'xytext': (dt.datetime(2019, 1, 30), 165000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2019, 5, 4), 190000),
                     'xytext': (dt.datetime(2019, 4, 4), 220000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2019, 6, 6), 180000),
                     'xytext': (dt.datetime(2019, 5, 6), 210000),
                     'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Peak season', 'xy': (dt.datetime(2019, 7, 26), 310000),
                     'xytext': (dt.datetime(2019, 6, 12), 280000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2019, 9, 13), 205000),
                     'xytext': (dt.datetime(2019, 9, 23), 235000),
                    'arrowprops': dict(facecolor='darkblue', shrink=0.05)},
                    {'text': 'Holidays', 'xy': (dt.datetime(2019, 12, 24), 190000),
                     'xytext': (dt.datetime(2019, 11, 20), 220000),
                     'arrowprops': dict(facecolor='darkblue', shrink=0.05)}]

prep.draw_plot_with_hline(df=res_17_grp_by_rent_day,
                          line_feat='sales', line_col='k', line_label='Sales',
                          h_line_val=res_17_sales_mean, hline_col='red', hline_label='Average Sales',
                          xlabel='Rental Day', ylabel='Sales (won)', title='2017 Sales', annot=sales_annot_2017)

prep.draw_plot_with_hline(df=res_18_grp_by_rent_day,
                          line_feat='sales', line_col='k', line_label='Sales',
                          h_line_val=res_18_sales_mean, hline_col='red', hline_label='Average Sales',
                          xlabel='Rental Day', ylabel='Sales (won)', title='2018 Sales', annot=sales_annot_2018)

prep.draw_plot_with_hline(df=res_19_grp_by_rent_day.groupby('rent_day').mean(),
                          line_feat='sales', line_col='k', line_label='Sales',
                          h_line_val=res_19_sales_mean, hline_col='red', hline_label='Average Sales',
                          xlabel='Rental Day', ylabel='Sales (won)', title='2019 Sales', annot=sales_annot_2019)

# Correlation of lead time & sales
res_17_grp_by_rent_day[['lead_time', 'sales']].corr()  # 0.866
res_18_grp_by_rent_day[['lead_time', 'sales']].corr()  # 0.827
res_19_grp_by_rent_day[['lead_time', 'sales']].corr()  # 0.720

res_17_grp_by_rent_day_scaled = prep.scaler(df=res_17_grp_by_rent_day, method='mnmx')
res_18_grp_by_rent_day_scaled = prep.scaler(df=res_18_grp_by_rent_day, method='mnmx')
res_19_grp_by_rent_day_scaled = prep.scaler(df=res_19_grp_by_rent_day, method='mnmx')

# Correlation Plot

# 2017 year
fig, axes = plt.subplots(1, 1)
res_17_grp_by_rent_day_scaled['lead_time'].plot.line(figsize=(15, 6), linewidth=0.8, alpha=0.7,
                                                     color='crimson', label='Lead Time', ax=axes)
res_17_grp_by_rent_day_scaled['sales'].plot.line(figsize=(15, 6), linewidth=0.8, alpha=0.9,
                                                 color='slateblue', label='Sales', ax=axes)
axes.legend()
axes.text(dt.datetime(2017, 11, 17), 0.82, 'Corr: 0.866', style='italic', color='dimgray')
title = "2017 Lead Time & Sales Correlation"
plt.title(title)
plt.savefig(os.path.join('.', 'img', title + '.png'))

# 2018 year
fig, axes = plt.subplots(1, 1)
res_18_grp_by_rent_day_scaled['lead_time'].plot.line(figsize=(15, 6), linewidth=0.8, alpha=0.7,
                                                     color='crimson', label='Lead Time', ax=axes)
res_18_grp_by_rent_day_scaled['sales'].plot.line(figsize=(15, 6), linewidth=0.8, alpha=0.9,
                                                 color='slateblue', label='Sales', ax=axes)
axes.legend()
axes.text(dt.datetime(2018, 11, 17), 0.82, 'Corr: 0.827', style='italic', color='dimgray')
title = "2018 Lead Time & Sales Correlation"
plt.title(title)
plt.savefig(os.path.join('.', 'img', title + '.png'))

# 2019 year
fig, axes = plt.subplots(1, 1)
res_19_grp_by_rent_day_scaled['lead_time'].plot.line(figsize=(15,6), linewidth=0.8, alpha=0.7,
                                                     color='crimson', label='Lead Time', ax=axes)
res_19_grp_by_rent_day_scaled['sales'].plot.line(figsize=(15,6), linewidth=0.8, alpha=0.9,
                                                 color='slateblue', label='Sales', ax=axes)
axes.legend()
axes.text(dt.datetime(2019, 11, 17), 0.82, 'Corr: 0.720',
          style='italic', color='dimgray')
title = "2019 Lead Time & Sales Correlation"
plt.title(title)
plt.savefig(os.path.join('.', 'img', title + '.png'))

print("")
