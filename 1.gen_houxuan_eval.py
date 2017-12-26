# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/16 11:16'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '1.gen_houxuan_eval.py'

import pandas as pd
from collections import defaultdict
import numpy as np
import os
import pickle as pk
import dill

cur_file_path = os.path.dirname(os.path.realpath(__file__))
EVALUATION = os.path.join(cur_file_path, '..', 'data', 'evaluation_public.csv')
HOUXUAN_EVAL = os.path.join(cur_file_path, '..', 'data', 'tmp', 'HOUXUAN_EVAL.pkl')
WIFI_SHOP_DIC = os.path.join(cur_file_path, '..', 'data', 'tmp', 'WIFI_SHOP_DIC.pkl')


def trans(line):
    res = ''
    for k, v in line.items():
        res += str(k) + ':' + str(v) + '|'
    return res[:-1]


def read_raw_data():
    if os.path.exists(HOUXUAN_EVAL):
        print('EVALUATION exist and is read at： ', HOUXUAN_EVAL)
        evaluation = pk.load(open(HOUXUAN_EVAL, 'rb'))
    else:
        print('starting load wifi shops dict...')
        wifi_to_shops = dill.load(open(WIFI_SHOP_DIC, 'rb'))
        print('starting read raw data and do some simple trans')
        evaluation = pd.read_csv(EVALUATION, date_parser=['time_stamp'])
        del evaluation['user_id']
        evaluation = evaluation.drop('wifi_infos', axis=1).join(
            evaluation['wifi_infos'].str.split(';', expand=True).stack().
                reset_index(level=1, drop=True).rename('wifi_infos'))
        evaluation['wifi_id'] = evaluation['wifi_infos'].apply(lambda x: x.split('|')[0])
        evaluation['wifi_signal'] = evaluation['wifi_infos'].apply(lambda x: x.split('|')[1])
        evaluation['wifi_conn'] = evaluation['wifi_infos'].apply(lambda x: x.split('|')[2])
        del evaluation['wifi_infos']
        evaluation.sort_values('wifi_signal', inplace=True)
        evaluation = evaluation.groupby(['row_id', 'time_stamp', 'longitude']).head(4)  # 取出信号最强的四个

        print('start generate hou xuan evaluation set...')
        evaluation['top_shops_from_wifi'] = evaluation['wifi_id'].apply(lambda x: trans(wifi_to_shops[x]))
        evaluation = evaluation.drop('top_shops_from_wifi', axis=1).join(evaluation['top_shops_from_wifi'].
                                                                         str.split('|', expand=True).stack().
                                                                         reset_index(level=1, drop=True).
                                                                         rename('shop_id_add'))
        evaluation['shop_id_1'] = evaluation['shop_id_add'].apply(lambda x: x.split(':')[0])
        evaluation['shop_id_add_number'] = evaluation['shop_id_add'].apply(
            lambda x: x.split(':')[1] if x and len(x) > 1 else 0)
        del evaluation['shop_id_add']
        evaluation['label'] = np.nan
        pk.dump(evaluation, open(HOUXUAN_EVAL, 'wb'))
        print('hou xuan evaluation set is generated and is saved at: ', HOUXUAN_EVAL)
    return evaluation


if __name__ == '__main__':
    evaluation = read_raw_data()
    print(evaluation.head())
'''
   row_id mall_id        time_stamp   longitude   latitude     wifi_id  \
0  118742  m_3916  2017-09-05 13:00  122.141011  39.818847  b_37756289   
0  118742  m_3916  2017-09-05 13:00  122.141011  39.818847  b_37756289   
0  118742  m_3916  2017-09-05 13:00  122.141011  39.818847  b_37756289   
0  118742  m_3916  2017-09-05 13:00  122.141011  39.818847  b_37756289   
0  118742  m_3916  2017-09-05 13:00  122.141011  39.818847  b_37756289   

  wifi_signal wifi_conn  shop_id_1 shop_id_add_number  label  
0         -53     false   s_147921                292    NaN  
0         -53     false  s_3683488                 49    NaN  
0         -53     false   s_312218                140    NaN  
0         -53     false    s_10425                 20    NaN  
0         -53     false  s_3683488                 56    NaN  
'''
