#!/usr/bin/python
#-*- coding:utF-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb

# read data 
na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']
data_set = pd.read_table('/home/fsg/jiwenchao/ac_class/data/data_set_0705.txt',sep = '\t',na_values = na_values)


# split train valid oot dataset 
train_set = data_set[(data_set.data_set_tag == 1)]
valid_set = data_set[(data_set.data_set_tag == 2)]
oot_set = data_set[(data_set.data_set_tag == 3)]

# selected model vars 
col_x_old = list(data_set.columns)

#只保留所需X变量
remove_vars=['passid','sessionid','data_set_tag','risk_tag_union']
col_x = [x for x in col_x_old if x not in remove_vars]

# to DMatrix
train_data = train_set[col_x].as_matrix()
train_label = train_set['risk_tag_union'].as_matrix()
dtrain = xgb.DMatrix(train_data, label = train_label)

valid_data = valid_set[col_x].as_matrix()
valid_label = valid_set['risk_tag_union'].as_matrix()
dvalid = xgb.DMatrix(valid_data, label = valid_label)

oot_data = oot_set[col_x].as_matrix()
oot_label = oot_set['risk_tag_union'].as_matrix()
doot = xgb.DMatrix(oot_data, label = oot_label)

# 定义验证数据集
evallist = [(dtrain, 'train'), (dvalid, 'valid'), (doot, 'oot')]

