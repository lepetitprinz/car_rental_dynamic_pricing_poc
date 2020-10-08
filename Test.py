from DataHandler import DataHandler
from Preprocessing import Preprocessing

prep = Preprocessing()

df = prep.load_data(path='test.csv', format='csv')

print("")