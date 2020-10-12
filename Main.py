# from DataHandler import DataHandler
from Preprocessing import Preprocessing

import os
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

# View histogram: count of discount change
prep.draw_histogram(df=discount, feature='chng_cnt')

# View line plot: count of discount change
prep.draw_line_plot(df=discount, x='date', y='chng_cnt')

# View monthly average count plot
prep.draw_plot_by_resample(df=discount, feature='chng_cnt', agg='mean', rule='M')

# Reservation dataset
res_17 = prep.load_data(path='reservation_17.csv', file_format='csv', delimiter='\t')
res_18 = prep.load_data(path='reservation_17.csv', file_format='csv', delimiter='\t')
res_19 = prep.load_data(path='reservation_17.csv', file_format='csv', delimiter='\t')

# Merge 17 ~ 19 years reservation dataset
res = pd.concat([res_17, res_18, res_19], axis=0)
