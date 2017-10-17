#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb

# 定义验证数据集
evallist = [(dtrain, 'train'), (dvalid, 'valid'), (doot, 'oot')]

# parameters
param = { 'objective': 'binary:logistic',    #定义学习任务及相应的学习目标
          'eval_metric': 'auc',      #校验数据所需要的评价指标
          'max_depth': 3,            #避免过拟合，过大则会学到更多的局部特征，易导致过拟合
          'learning_rate': 0.01,     #可用来防止过拟合，eta，学习速率，更新过程中用到的收缩步长
          'min_child_weight': 50,   #子节点中最小的样本权重和，调高可以避免过拟合，越大算法越conservative
          'silent': 1,               #取1时表示以缄默方式运行，不打印运行时信息，取0时表示打印出运行时信息。
          'lambda':1,               #L2正则化权重，减少过拟合
          'alpha': 1,               #L1正则化权重，减少过拟合 
          'gamma': 0.8,             #值越大，算法越保守
          'max_delta_step': 1,       #限制每棵树权重改变的最大步长
          'subsample': 0.95,         #避免过拟合，但是如果过小，则易导致欠拟合，子样本占整个样本集合的比例，可防止过拟合
          'colsample_bytree': 0.95,  #避免过拟合，但是如果过小，则易导致欠拟合，在建立树时对特征采样的比例，可防止过拟合
          'scale_pos_weight': 1,    #类别十分不平衡
          'seed': 1                  #随机数的种子。缺省值为0
          }

# set the max number of iteration
num_rounds=950

#tunning the model
bst = xgb.train(param,dtrain,num_boost_round = num_rounds , evals = evallist , early_stopping_rounds = 10 )


# if early stopping is enabled during training, you can get predictions from the best iteration with bst.best_ntree_limit
bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
