import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from urllib.request import urlopen

class DataHandler(object):
    # -------------------------------------------- #
    # Define Functions
    # -------------------------------------------- #

    def get_xml_data(self, xtree):
        rows = []
        for node in xtree[1][0]:
            airport = node.find("airport").text
            arrflgt = node.find("arrflgt").text
            arrpassenger = node.find("arrpassenger").text
            depflgt = node.find("depflgt").text
            deppassenger = node.find("deppassenger").text
            subflgt = node.find("subflgt").text
            subpassenger = node.find("subpassenger").text

            rows.append({"airport": airport, "arrflgt": arrflgt, "arrpassenger": arrpassenger,
                         "arrpassenger": arrpassenger, "depflgt": depflgt, "deppassenger": deppassenger,
                         "subflgt": subflgt, "subpassenger": subpassenger})

        return rows

    def get_airport_data(self, date_list: list, ulr: str, service_key: str):

        for date in date_list:
            # url_opt = f'?startDePd={date[0]}&endDePd={date[1]}&serviceKey={service_key}'
            url_opt = f'?startDePd={date[0]}&endDePd={date[1]}&routeBe=1&pasngrCargoBe=0&serviceKey={service_key}'
            url_fin = ulr + url_opt
            response = urlopen(url_fin).read()
            xtree = ET.fromstring(response)

            rows = self.get_xml_data(xtree=xtree)

        return rows

    def get_prep_df(rows: list, format: int):
        if format == 1:
            temp = pd.DataFrame(rows)
            temp = temp[temp['ageCode'] > '20']  # 20세 이상
            temp = temp[temp['ageCode'] != '99']  # 승무원 제외
            temp = temp[(temp['portCode'] == 'IA') | (temp['portCode'] == 'GP') | (temp['portCode'] == 'CJ')]
            temp['num'] = temp['num'].astype(np.int64)

        else:
            temp = pd.DataFrame(rows)
            temp['average'] = temp['average'].astype(np.float64)

        return temp

    def get_stats_df(yyyymm_list: list, url: str, format: int, num_of_row: int, servicekey: str):

        nums = []

        for yyyymm in yyyymm_list:
            url_opt = f'?YM={yyyymm}&numOfRows={num_of_row}&serviceKey={servicekey}'
            url_fin = url + url_opt
            response = urlopen(url_fin).read()
            xtree = ET.fromstring(response)

            rows = DataHandler.get_xml_data(xtree=xtree, format=format)
            temp = DataHandler.get_prep_df(rows=rows, format=format)
            if format == 1:
                nums.append({'yyyymm': yyyymm, 'num_of_people': temp['num'].sum()})
            else:
                nums.append({'yyyymm': yyyymm, 'avg_retention_time': round(temp['average'].mean())})

        df = pd.DataFrame(nums)
        if format == 1:
            df = df[['yyyymm', 'num_of_people']]
        else:
            df = df[['yyyymm', 'avg_retention_time']]
        return df

# -------------------------------------------- #
# Main
# -------------------------------------------- #
data_handler = DataHandler()

url = 'http://openapi.airport.co.kr/service/rest/totalAirportStatsService/getAirportStats'
service_key = 'UYRNns1wVRWz8MIyaMqUcL%2BHhIsbY0xjNyzRyvBNZRwh9zefraNj4lh9eBLgOw%2B2c8lBV%2Fh1SbzyNV96aO3DUw%3D%3D'

data = data_handler.get_airport_data(date_list=[(201901, 201901)], ulr=url, service_key=service_key)
print('')