from DataHandler import DataHandler
from Preprocessing import Preprocessing

prep = Preprocessing()

df = prep.load_data(path='jeju.xls', format='excel')

print("")