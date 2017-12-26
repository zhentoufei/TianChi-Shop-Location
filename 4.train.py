# -*- coding: utf-8 -*-
__author__ = 'Mr.Finger'
__date__ = '2017/11/16 16:10'
__site__ = ''
__software__ = 'PyCharm'
__file__ = '4.train.py'

import pandas as pd
import os
import pickle as pk
from sklearn import preprocessing
import lightgbm as lgb
# import xgboost as xgb
from xgboost import XGBClassifier
import gc

cur_file_path= os.path.dirname(os.path.dirname(__file__))
TRAIN_SET = os.path.join(cur_file_path, '..', 'data', 'tmp', 'train_set.pkl')
TRAIN_SET_ENCODING = os.path.join(cur_file_path, '..', 'data', 'tmp','train_set_encoding.pkl')

def encoding():
    print('start encoding the data set...')
    if os.path.exists(TRAIN_SET_ENCODING):
        return pk.load(open(TRAIN_SET_ENCODING, 'rb'))
    else:
        train_set = pk.load(open(TRAIN_SET, 'rb'))
        for col in train_set.columns:
            if train_set[col].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(train_set[col].values))
                train_set[col] = lbl.transform(list(train_set[col].values))
        pk.dump(train_set, open(TRAIN_SET_ENCODING, 'wb'))
        return train_set


def split_train_set(train_set):
    print('start split the train set...')
    exclude_features = ['time_stamp']
    features = [x for x in train_set.columns if x not in exclude_features]
    train_set['time_stamp'] = pd.to_datetime(train_set['time_stamp'])
    train = train_set[train_set['time_stamp'] < '2017-08-25']
    train = train[features]

    evaluation = train_set[train_set['time_stamp'] >= '2017-08-25']
    evaluation = evaluation[features]
    return train, evaluation

def train_lgb():
    train_set, evaluation_set = split_train_set(encoding())
    # train_set, evaluation_set = split_train_set()

    print('prepare for the training...')
    features = [x for x in train_set.columns if x not in ['label']]

    y_train = train_set['label'].values
    X_train = train_set[features].values

    y_test = evaluation_set['label'].values
    X_test = evaluation_set[features].values



    print(X_train)
    lgb_train = lgb.Dataset(X_train, y_train)
    print(lgb_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 256,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=180,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=15)

    result_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print(result_test)

def train_xgb():
    import xgboost as xgb
    train_set, evaluation_set = split_train_set(encoding())
    # train_set, evaluation_set = split_train_set()
    train_set.fillna(0, inplace=True)
    print(train_set.head())
    print('prepare for the training...')
    features = [x for x in train_set.columns if x not in ['label']]
    y_train = train_set['label']
    X_train = train_set[features]

    y_test = evaluation_set['label']
    X_test = evaluation_set[features]


    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_test = xgb.DMatrix(X_test)
    print('X_train shape')
    print(X_train.shape)
    print('y_train shape')
    print(y_train.shape)

    params = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'colsample_bytree': 0.886,
        'min_child_weight': 2,
        'max_depth': 10,
        'subsample': 0.886,
        # 'alpha': 10,
        # 'gamma': 30,
        # 'lambda': 50,
        'verbose_eval': True,
        'eval_metric': 'auc',
        'scale_pos_weight': 10,
        'seed': 201703,
        'missing': -1
    }
    xgb = xgb.train(params, xgb_train, early_stopping_rounds=20)
    pre = xgb.predict(xgb_test)
    print(type(pre))



if __name__ == '__main__':
    train_xgb()
























