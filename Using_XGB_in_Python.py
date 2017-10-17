#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import gc
import re

# 解决结果集中文显示问题
reload(sys)
sys.setdefaultencoding('utf8')

# recording system runing time
import time 
start_time = time.time()


# read data 
na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']
data_set = pd.read_table('./data_set_0705.txt',sep = '\t',na_values = na_values)


# split train valid oot dataset 
train_set = data_set[(data_set.data_set_tag == 1)]
valid_set = data_set[(data_set.data_set_tag == 2)]
oot_set = data_set[(data_set.data_set_tag == 3)]

# selected model vars 
col_x_old = list(data_set.columns)

#临时删除关键字段
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


#模型数据分配
evallist = [(dtrain, 'train'), (dvalid, 'valid'), (doot, 'oot')]

# parameters
param = { 'objective': 'binary:logistic',    #定义学习任务及相应的学习目标
          'eval_metric': 'auc',      #校验数据所需要的评价指标
          'max_depth': 3,            #避免过拟合，过大则会学到更多的局部特征，易导致过拟合
          'learning_rate': 0.01,     #可用来防止过拟合，eta，学习速率，更新过程中用到的收缩步长
          'min_child_weight': 50,    #子节点中最小的样本权重和，调高可以避免过拟合，越大算法越conservative
          'silent': 1,               #取1时表示以缄默方式运行，不打印运行时信息，取0时表示打印出运行时信息。
          'lambda':1,                #L2正则化权重，减少过拟合
          'alpha': 1,                #L1正则化权重，减少过拟合 
          'gamma': 0.8,              #值越大，算法越保守
          'max_delta_step': 1,       #限制每棵树权重改变的最大步长
          'subsample': 0.95,         #避免过拟合，但是如果过小，则易导致欠拟合，子样本占整个样本集合的比例，可防止过拟合
          'colsample_bytree': 0.95,  #避免过拟合，但是如果过小，则易导致欠拟合，在建立树时对特征采样的比例，可防止过拟合
          'scale_pos_weight': 1,     #类别十分不平衡
          'seed': 1                  #随机数的种子。缺省值为0
          }


# 迭代次数 the max number of iterations
num_rounds=950

#tunning the model
bst = xgb.train(param,dtrain,num_boost_round = num_rounds , evals = evallist)


print "best best_ntree_limit:",bst.best_ntree_limit 

#输出运行时间
endtime = time.time()
cost_time = endtime - start_time
print "xgboost success!",'\n',"cost time:",cost_time,"(s)"

#1、save model 
bst.save_model('./01_ac_class_fraud.model')

# dump model with feature map
feat_map_df = pd.DataFrame({'id': [i for i in range(len(col_x))]})
feat_map_df = feat_map_df.assign(feat_name = col_x)
feat_map_df = feat_map_df.assign(type = ['q' for i in range(len(col_x))])
feat_map_df.to_csv('./feat_map_ac_class_fraud.txt',sep = '\t',header = False,index = False,encoding='utf-8')

# 中文乱码解决方案
# reload(sys)
# sys.setdefaultencoding('utf8')
bst.dump_model('./feat_map_ac_class_fraud.dump.raw.txt', './feat_map_ac_class_fraud.txt')


## 输出重要变量
# 变量名称以及重要性得分字典表
fscore_dict = bst.get_fscore(fmap = './feat_map_ac_class_fraud.txt')

# 字典表转化为数据框的形式，包括key，value和value的标准化%
features = []
scores = []
for key in fscore_dict:
     features.append(key)
     scores.append(fscore_dict[key])

#重要性归一化，保留三位有效数字
ratio=[]
for key in fscore_dict:
     ratio.append(round(fscore_dict[key]*1.0/sum(scores)*100.0,3))

# 组合成数据框的形式，并按照score排序
fscore_df = pd.DataFrame({'features': features, 'scores': scores, 'ratio(%)':ratio})
fscore_df=fscore_df.sort_values(by=['scores'],ascending=False,inplace=False)

# 输出到本地txt文件
fscore_df.to_csv('./feature_importance_ac_class_fraud.txt',sep = '\t', header = True, index = False,encoding = 'utf-8')


## prediction
# read data 
na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']
data_set = pd.read_table('/home/fsg/jiwenchao/ac_class/data/data_set_0705.txt',sep = '\t',na_values = na_values)

# selected model vars 
col_x_old = list(data_set.columns)

#临时删除关联字段
remove_vars=['passid','sessionid','data_set_tag','risk_tag_union']
col_x = [x for x in col_x_old if x not in remove_vars]

# to DMatrix
data_set_all = data_set[col_x].as_matrix()
data_prob_value = xgb.DMatrix(data_set_all)

# load model 
bst = xgb.Booster()
bst.load_model('./01_ac_class_fraud.model')

#保留三列对应值
pred_df = data_set[['sessionid','passid','risk_tag_union']]
pred_df['score'] = bst.predict(data_prob_value)  #添加一列score，注意ddafen数据的类型，DMatrix


#保留数据，用于模型分修正
pred_df.to_csv('./data_set_score.txt',sep = '\t', header = True, index = False,encoding = 'utf-8')


## 修正参数
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression

# compute xbeta_old
xbeta_old = []
for i in range(pred_df.shape[0]):
     score_i = pred_df.score[i]
     if score_i == 1:
          xbeta_old.append([10])
     elif score_i == 0:
          xbeta_old.append([-10]) 
     else:
          xbeta_old.append([math.log(score_i/(1-score_i))])

tag = np.array(pred_df.risk_tag_union)

# use weight to restore sample
xbeta_old_w = []
tag_w = []
weight=219
for i in range(len(tag)):
     if tag[i] == 0:
          xbeta_old_w.extend([xbeta_old[i]] * weight)
          tag_w.extend([tag[i]] * weight)
     else:
          xbeta_old_w.append(xbeta_old[i])
          tag_w.append(tag[i])

# compute coef
lr = LogisticRegression()
lr.fit(xbeta_old_w, tag_w)
a = lr.coef_[0][0]
b = lr.intercept_[0]


# 模型分修正参数
a,b


pred_df = pred_df.assign(xbeta_old = np.log(pred_df.score/(1-pred_df.score)))
pred_df.loc[pred_df.xbeta_old >= 10, 'xbeta_old'] = 10 
pred_df.loc[pred_df.xbeta_old <= -10, 'xbeta_old'] = -10
pred_df = pred_df.assign(xbeta_new = pred_df.xbeta_old*a + b)
pred_df = pred_df.assign(score_new = 1/(1+np.exp(-1 * pred_df.xbeta_new)))

# 保存修正后的数据
pred_df.to_csv('./ac_class_score_new.txt',sep = '\t',header = True,index = False)


## 验证修正是否正确。注意扩展抽样数据

data=pred_df
#正负样本分离
data_risk_tag_union_0=data[data.risk_tag_union==0]
data_risk_tag_union_1=data[data.risk_tag_union==1]

#正样本扩展，weight=219
data_risk_tag_union_0_all=pd.concat([data_risk_tag_union_0]*219)

#上下拼接扩展后的正样本和全部负样本
data=pd.concat([data_risk_tag_union_0_all,data_risk_tag_union_1])


#验证是否修正后的均值是否一致
data.score_new.describe()

#导出数据用于cutoff
data[['sessionid', 'passid','risk_tag_union','score_new']].to_csv(xpath + '/ac_class_score_new_last.txt',sep = '\t',header = True,index = False)


###############################################################################  END  ############################################################################


# 生成pmml文件模板
java -jar target/converter-executable-1.2-SNAPSHOT.jar --model-input xgboost.model --fmap-input xgboost.fmap --target-name mpg --pmml-output xgboost.pmml
