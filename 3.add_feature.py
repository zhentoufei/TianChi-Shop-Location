# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/16 14:33'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '3.add_feature.py'

import pandas as pd
from collections import defaultdict
import os
import pickle as pk
import numpy as np
import gc
from sklearn import preprocessing
from sklearn.decomposition import PCA

cur_file_path = os.path.dirname(os.path.realpath(__file__))
HOUXUAN_TRAIN = os.path.join(cur_file_path, '..', 'data', 'tmp', 'HOUXUAN_TRAIN.pkl')
WIFI_ID_W2V_DATAFRAME = os.path.join(cur_file_path, '..', 'data', 'tmp', 'WIFI_ID_W2V_DATAFRAME.pkl')
SHOP_INFO = os.path.join(cur_file_path, '..', 'data', 'ccf_first_round_shop_info.csv')


def load_train_set():
    print('start load train set pickle...')
    train_set = pk.load(open(HOUXUAN_TRAIN, 'rb'))
    return train_set


def load_wifi_id_w2v():
    print('start load wifi id w2v pickle...')
    wifi_id_w2v = pk.load(open(WIFI_ID_W2V_DATAFRAME, 'rb'))
    return wifi_id_w2v


def train_set_add_wifi_frature():
    '''
    加入word2vec的特征
    :return:
    '''
    train_set = load_train_set()
    wifi_id_w2v = load_wifi_id_w2v()
    not_required_features = ['key_0']
    required_features = [x for x in wifi_id_w2v.columns if x not in not_required_features]

    wifi_id_col = wifi_id_w2v['key_0']
    pca = PCA(n_components=5)
    pca.fit(wifi_id_w2v[required_features].values)
    wifi_id_w2v = pca.transform(wifi_id_w2v[required_features].values)
    wifi_id_w2v = pd.concat([wifi_id_col, pd.DataFrame(wifi_id_w2v)], axis=1)

    print('start add wifi infos...')
    # result.rename(columns={'geohashed_start_loc': 'geohashed_end_loc'}, inplace=True)
    wifi_id_w2v.rename(columns={'key_0': 'wifi_id'}, inplace=True)
    wifi_id_w2v.rename(columns={0: 'w2v_fea_0'}, inplace=True)
    wifi_id_w2v.rename(columns={1: 'w2v_fea_1'}, inplace=True)
    wifi_id_w2v.rename(columns={2: 'w2v_fea_2'}, inplace=True)
    wifi_id_w2v.rename(columns={3: 'w2v_fea_3'}, inplace=True)
    wifi_id_w2v.rename(columns={4: 'w2v_fea_4'}, inplace=True)
    train_set = pd.merge(train_set, wifi_id_w2v, on=['wifi_id'], how='left')
    exclude_fea = ['index', 'shop_id', 'wifi_id', 'shop_id_1']
    features = [x for x in train_set.columns if x not in exclude_fea]
    train_set = train_set[features]
    return train_set


def train_set_trans_conn(train):
    print('start transform conn infos...')
    train['wifi_conn'] = train['wifi_conn'].apply(lambda x: 0 if str(x) == 'false' else 1)
    return train


def train_set_add_time_fea(train):
    print('start add time infos...')
    train['day'] = pd.DatetimeIndex(train['time_stamp']).dayofweek
    train['day'] = train['day'].apply(lambda x: 0 if x >= 0 and x <= 4 else 1)
    train['hour'] = pd.DatetimeIndex(train['time_stamp']).hour
    return train


def train_set_add_mall_category(train):
    print('start add mall cate infos...')
    shop_infos = pd.read_csv(SHOP_INFO)
    mall_cate = defaultdict(lambda : set())
    cate_id = list(shop_infos.columns).index('category_id')
    mall_id = list(shop_infos.columns).index('mall_id')
    for line in shop_infos.values:
        mall_cate[line[mall_id]].add(line[cate_id])

    mall_cate_nums = {}
    for k, v in mall_cate.items():
        mall_cate_nums[k] = len(v)

    train['cate_nums'] = train['mall_id'].apply(lambda x: mall_cate_nums[x])
    return train

if __name__ == '__main__':
    train = train_set_add_wifi_frature()
    train = train_set_trans_conn(train)
    train = train_set_add_time_fea(train)
    train = train_set_add_mall_category(train)
    path = os.path.join(cur_file_path, '..', 'data', 'tmp', 'train_set.pkl')
    pk.dump(train, open(path, 'wb'))
    print(train.head())































