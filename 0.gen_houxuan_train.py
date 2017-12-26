# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/16 10:58'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '0.gen_houxuan_train.py'

import pandas as pd
from collections import defaultdict
import os
import pickle as pk
import dill
import numpy as np

cur_file_path = os.path.dirname(os.path.realpath(__file__))
os.path.join(cur_file_path, "..", "models", "sentence_tfidf_wcd.bin")
USER_SHOP_BEHAVIOR = os.path.join(cur_file_path, '..', 'data', 'ccf_first_round_user_shop_behavior.csv')
SHOP_INFO = os.path.join(cur_file_path, '..', 'data', 'ccf_first_round_shop_info.csv')
HOUXUAN_TRAIN = os.path.join(cur_file_path, '..', 'data', 'tmp', 'HOUXUAN_TRAIN.pkl')
WIFI_SHOP_DIC = os.path.join(cur_file_path, '..', 'data', 'tmp', 'WIFI_SHOP_DIC.pkl')


def read_raw_train():
    print('starting reading raw data and do something simple trans...')
    user_shop_behav = pd.read_csv(USER_SHOP_BEHAVIOR, date_parser=['time_stamp'])
    shop_info = pd.read_csv(SHOP_INFO, date_parser=['time_stamp'])
    user_shop_behav = pd.merge(user_shop_behav, shop_info[['shop_id', 'mall_id']], how='left', on='shop_id')
    user_shop_behav.rename(columns={'longitude_x': 'longitude'}, inplace=True)
    user_shop_behav.rename(columns={'latitude_x': 'latitude'}, inplace=True)
    del user_shop_behav['user_id']
    user_shop_behav = user_shop_behav.drop('wifi_infos', axis=1).join(user_shop_behav['wifi_infos'].
                                                                      str.split(';', expand=True).stack().
                                                                      reset_index(level=1, drop=True).
                                                                      rename('wifi_infos'))
    user_shop_behav = user_shop_behav.reset_index()
    user_shop_behav['wifi_id'] = user_shop_behav['wifi_infos'].apply(lambda x: x.split('|')[0])
    user_shop_behav['wifi_signal'] = user_shop_behav['wifi_infos'].apply(lambda x: x.split('|')[1])
    user_shop_behav['wifi_conn'] = user_shop_behav['wifi_infos'].apply(lambda x: x.split('|')[2])
    del user_shop_behav['wifi_infos']

    user_shop_behav.sort_values('wifi_signal', inplace=True)
    user_shop_behav = user_shop_behav.groupby(['shop_id', 'time_stamp', 'longitude']).head(4)  # 取出信号最强的四个
    return user_shop_behav


# 在数据变换的时候要使用
def trans(line):
    res = ''
    for k, v in line.items():
        res += str(k) + ':' + str(v) + '|'
    return res[:-1]


def gen_houxuan(train):
    print('starting gen wifi shop dict...')
    if os.path.exists(WIFI_SHOP_DIC):
        print('WIFI_SHOP_DIC exists and is read at: ', WIFI_SHOP_DIC)
        wifi_to_shops = dill.load(open(WIFI_SHOP_DIC, 'rb'))
    else:
        print('starting generate the wifi shop dict...')
        wifi_to_shops = defaultdict(lambda: defaultdict(lambda: 0))
        cols = list(train.columns)
        wifi_id = cols.index('wifi_id')
        shop_id = cols.index('shop_id')
        # 获得所有的shop_id和wifi_id的键值对的个数
        for line in train.values:
            wifi_to_shops[line[wifi_id]][line[shop_id]] += 1
        dill.dump(wifi_to_shops, open(WIFI_SHOP_DIC, 'wb'))

    if os.path.exists(HOUXUAN_TRAIN):
        print('HOUXUAN_TRAIN exists and is read at: ', HOUXUAN_TRAIN)
        train = pk.load(open(HOUXUAN_TRAIN, 'rb'))
    else:
        print('start generate hou xuan train set...')
        train['top_shops_from_wifi'] = train['wifi_id'].apply(lambda x: trans(wifi_to_shops[x]))
        train = train.drop('top_shops_from_wifi', axis=1).join(train['top_shops_from_wifi'].
                                                               str.split('|', expand=True).stack().
                                                               reset_index(level=1, drop=True).
                                                               rename('shop_id_add'))
        train['shop_id_1'] = train['shop_id_add'].apply(lambda x: x.split(':')[0])
        train['shop_id_add_number'] = train['shop_id_add'].apply(lambda x: x.split(':')[1])
        del train['shop_id_add']
        train['label'] = (train['shop_id'] == train['shop_id_1']).astype(int)
        pk.dump(train, open(HOUXUAN_TRAIN, 'wb'))
        print('hou xuan train set is generated and is saved at: ', HOUXUAN_TRAIN)
    return train


if __name__ == '__main__':
    user_shop_behav = read_raw_train()
    train = gen_houxuan(user_shop_behav)
    print(train.head())
'''
   index    shop_id        time_stamp   longitude  latitude mall_id  \
0      0  s_2871718  2017-08-06 21:20  122.308291  32.08804  m_1409   
0      0  s_2871718  2017-08-06 21:20  122.308291  32.08804  m_1409   
0      0  s_2871718  2017-08-06 21:20  122.308291  32.08804  m_1409   
3      0  s_2871718  2017-08-06 21:20  122.308291  32.08804  m_1409   
6      0  s_2871718  2017-08-06 21:20  122.308291  32.08804  m_1409   

     wifi_id wifi_signal wifi_conn  shop_id_1 shop_id_add_number  label  
0  b_6396480         -67     false  s_2871718                 42      1  
0  b_6396480         -67     false   s_580441                  2      0  
0  b_6396480         -67     false  s_1877554                  3      0  
3  b_6396479         -55     false  s_2871718                 47      1  
6  b_5857370         -68     false  s_2871718                 60      1  
'''
