# from DataHandler import DataHandler
from Preprocessing import Preprocessing

import os
import copy
import pandas as pd

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
prep.draw_histogram(df=discount, feature='chng_cnt')

# View line plot: count of discount change
# prep.draw_line_plot(df=discount_17, x='date', y='chng_cnt', interval=1, title='2017 Discount Change')
# prep.draw_line_plot(df=discount_18, x='date', y='chng_cnt', interval=1, title='2018 Discount Change')
# prep.draw_line_plot(df=discount_19, x='date', y='chng_cnt', interval=1, title='2019 Discount Change')
# prep.draw_line_plot(df=discount_20, x='date', y='chng_cnt', interval=1, title='2020 Discount Change')

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
             '실대여기간(일)': 'real_rent_period', '차량대여요금(VAT포함)': 'car_rent_fee',
             'CDW가입여부': 'cdw_yn', 'CDW요금구분': 'cdw_fee_kind', 'CDW요금구분명': 'cdw_fee_kind_nm',
             'CDW요금': 'cdw_fee', '회원등급': 'member_grd','차종': 'car_kind', '구매목적': 'rent_purpose',
             '내부매출액': 'sales', '수납': 'pay_kind', '예약일자': 'rev_date', '할인유형': 'discount_kind',
             '할인유형명': 'discount_kind_nm', '적용할인명': 'apply_discount_nm'}

res.rename(columns=col_remap, inplace=True)

# Drop columns
drop_cols = ['tot_pay', 'tot_balance',
             'cdw_yn','cdw_fee_kind', 'cdw_fee_kind_nm', 'cdw_fee',
             'car_grd', 'car_kind', 'rent_purpose', 'pay_kind']

res = res.drop(columns=drop_cols, axis=1)
res = prep.conv_to_datetime(df=res, str_feat='rent_day', datetime_feat='rent_day', date_format='%Y-%m-%d')
res = prep.conv_to_datetime(df=res, str_feat='return_day', datetime_feat='return_day', date_format='%Y-%m-%d')
res = prep.conv_to_datetime(df=res, str_feat='rev_date', datetime_feat='rev_date', date_format='%Y-%m-%d')

# Calculate Lead Time
res['lead_time'] = res['rent_day'] - res['rev_date']
res['lead_time'] = res['lead_time'].apply(lambda x: x.days)
res = res.sort_values(by=['rent_day'])

# Seperate dataset by years
res_17 = copy.deepcopy(res[res['rent_day'] <= pd.to_datetime('20171231', format='%Y-%m-%d')])
res_18 = copy.deepcopy(res[(res['rent_day'] > pd.to_datetime('20171231', format='%Y-%m-%d')) &
                          (res['rent_day'] <= pd.to_datetime('20181231', format='%Y-%m-%d'))])
res_19 = copy.deepcopy(res[res['rent_day'] >= pd.to_datetime('20190101', format='%Y-%m-%d')])

# Mean of lead time each year
res_17_lead_time_mean = res_17['lead_time'].mean()
res_18_lead_time_mean = res_18['lead_time'].mean()
res_19_lead_time_mean = res_19['lead_time'].mean()

# Plot
prep.draw_plot_with_hline(df=res_17.groupby('rent_day').mean(),
                          line_feat='lead_time', line_col='k', line_label='Lead Time',
                          h_line_val=res_17_lead_time_mean, hline_col='red', hline_label='Average Lead Time',
                          xlabel='Rental Day', ylabel='Lead Time (days)', title='2017 Lead Time')

prep.draw_plot_with_hline(df=res_18.groupby('rent_day').mean(),
                          line_feat='lead_time', line_col='k', line_label='Lead Time',
                          h_line_val=res_18_lead_time_mean, hline_col='red', hline_label='Average Lead Time',
                          xlabel='Rental Day', ylabel='Lead Time (days)', title='2018 Lead Time')

prep.draw_plot_with_hline(df=res_19.groupby('rent_day').mean(),
                          line_feat='lead_time', line_col='k', line_label='Lead Time',
                          h_line_val=res_19_lead_time_mean, hline_col='red', hline_label='Average Lead Time',
                          xlabel='Rental Day', ylabel='Lead Time (days)', title='2019 Lead Time')

print("")