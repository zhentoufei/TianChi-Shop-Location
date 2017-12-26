# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/16 11:39'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '3.gen_wifi_id_w2v_features.py'

import pandas as pd
from collections import defaultdict
import os
import pickle as pk
from gensim.models import word2vec, KeyedVectors

cur_file_path = os.path.dirname(os.path.realpath(__file__))
os.path.join(cur_file_path, "..", "models", "sentence_tfidf_wcd.bin")
USER_SHOP_BEHAVIOR = os.path.join(cur_file_path, '..', 'data', 'ccf_first_round_user_shop_behavior.csv')
EVALUATION = os.path.join(cur_file_path, '..', 'data', 'evaluation_public.csv')
SHOP_INFO = os.path.join(cur_file_path, '..', 'data', 'ccf_first_round_shop_info.csv')
WIFI_ID_W2V_BIN = os.path.join(cur_file_path, '..', 'models', 'WIFI_ID_W2V_BIN_150.bin')
WIFI_ID_W2V_TXT = os.path.join(cur_file_path, '..', 'models', 'WIFI_ID_W2V_TXT_150.txt')
WIFI_ID_W2V_DATAFRAME = os.path.join(cur_file_path, '..', 'data', 'tmp', 'WIFI_ID_W2V_DATAFRAME.pkl')


def read_raw_train():
    print('starting read the raw train...')
    user_shop_behav = pd.read_csv(USER_SHOP_BEHAVIOR, date_parser=['time_stamp'])
    shop_info = pd.read_csv(SHOP_INFO, date_parser=['time_stamp'])
    user_shop_behav = pd.merge(user_shop_behav, shop_info[['shop_id', 'mall_id']], how='left', on='shop_id')
    user_shop_behav.rename(columns={'longitude_x': 'longitude'}, inplace=True)
    user_shop_behav.rename(columns={'latitude_x': 'latitude'}, inplace=True)
    del user_shop_behav['user_id']
    return user_shop_behav


def read_raw_eva():
    print('starting read the evaluation...')
    evaluation = pd.read_csv(EVALUATION, date_parser=['time_stamp'])
    del evaluation['user_id']
    return evaluation

def gen_list_of_list_of_word(train):
    print('start generate wifi id list of list...')
    list_of_list_of_word = []
    index_of_wifi_infos = list(train.columns).index('wifi_infos')
    for line in train.values:
        wifi_infos = line[index_of_wifi_infos].split(';')
        cur_info = []
        for wifi_info in wifi_infos:
            cur_info.extend(wifi_info.split('|'))
        list_of_list_of_word.append(cur_info)
    return list_of_list_of_word


def word2vec_train():
    if os.path.exists(WIFI_ID_W2V_BIN):
        print('WIFI_ID_W2V_BIN exists at: ', WIFI_ID_W2V_BIN)
        return
    else:
        user_shop_behav = read_raw_train()
        evaluation = read_raw_eva()
        behav_and_eva = pd.concat([user_shop_behav, evaluation])
        list_of_list_of_word = gen_list_of_list_of_word(behav_and_eva)
        print('start train word2vec models...')
        model = word2vec.Word2Vec(list_of_list_of_word, size=150, min_count=3, workers=4)
        model.wv.save_word2vec_format(WIFI_ID_W2V_BIN, binary=True)
        model.wv.save_word2vec_format(WIFI_ID_W2V_TXT, binary=False)
        print('WIFI_ID_W2V_BIN is saved at: ', WIFI_ID_W2V_BIN)

def gen_wifi_id_w2v_dataframe():
    if os.path.exists(WIFI_ID_W2V_DATAFRAME):
        print('WIFI_ID_W2V_DATAFRAME exists and is read at: ', WIFI_ID_W2V_DATAFRAME)
        res = pk.load(open(WIFI_ID_W2V_DATAFRAME, 'rb'))
    else:
        m = []
        with open(WIFI_ID_W2V_TXT, 'r') as file:
            for line in file:
                if line.startswith('b'):
                    tmp = line.split(' ')
                    dic = {}
                    for k, v in enumerate(tmp):
                        dic['key_' + str(k)] = v
                    m.append(dic)
        res = pd.DataFrame(m)
        pk.dump(res, open(WIFI_ID_W2V_DATAFRAME, 'wb'))
        print('WIFI_ID_W2V_DATAFRAME is generated and saved at: ', WIFI_ID_W2V_DATAFRAME)
    return res



if __name__ == '__main__':
    wifi_id_w2v_dataframe = gen_wifi_id_w2v_dataframe()
    print(wifi_id_w2v_dataframe.head())

'''
        key_0      key_1     key_10    key_100   key_101    key_102  \
0  b_33503892   0.663924  -0.427696  -2.236387  1.193954   1.169104   
1   b_4997636   0.500833   0.313217   0.692607  1.099157  -0.222078   
2  b_57104369  -0.485432  -0.523619  -1.582977  0.258449  -0.183902   
3   b_4997622   0.735973   0.487376   0.330111  1.225195   0.305935   
4   b_4997635   0.471485   0.452852   0.292835  0.813780   0.004700   

     key_103    key_104    key_105    key_106    ...         key_90  \
0  -0.781578  -1.810392  -0.111631  -1.183253    ...      -2.213830   
1  -1.390752   0.205334   0.058230   0.864334    ...      -0.419896   
2  -1.544221  -1.352750   0.123152   0.102904    ...      -0.626184   
3  -2.098470  -0.102615  -0.398526   0.867184    ...      -0.927213   
4  -1.824822  -0.042320  -0.585960   0.988571    ...      -0.737264   

      key_91     key_92     key_93     key_94     key_95    key_96     key_97  \
0   0.483542  -0.546117  -1.331233  -0.081697   0.182477  1.531183  -0.874296   
1  -0.077770   1.804042   0.315871  -0.560919  -0.137301  1.674913  -0.824791   
2  -0.085767   1.313736  -0.033135   0.141818  -0.259878  1.465369   0.782216   
3   0.227590   1.233988   0.564908  -0.725370  -0.581126  1.371480  -0.825959   
4   0.470610   1.726513   0.292988  -1.052749  -0.607663  1.979590   0.213512   

      key_98     key_99  
0   0.677866  -0.466803  
1  -0.308380  -1.139153  
2  -0.619330  -1.374338  
3  -0.184079  -0.843000  
4  -0.201508  -1.341005  

[5 rows x 151 columns]
'''