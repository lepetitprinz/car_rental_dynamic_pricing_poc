import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Preprocessing(object):
    ROOT_DIR: str = os.path.join('..', 'input')

    def __init__(self):
        self.check_dir

    def check_dir(self):
        if not os.path.exists(self.__class__.ROOT_DIR):
            os.mkdir(self.__class__.ROOT_DIR)

    def load_data(self, path: str, format: str):
        """
        :param format: csv / excel
        :return: dataframe
        """
        file_path = os.path.join(self.__class__.ROOT_DIR, path)
        if format == 'csv':
            return pd.read_csv(file_path)
        elif format == 'excel':
            return pd.read_excel(file_path)