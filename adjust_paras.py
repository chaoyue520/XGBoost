#!/usr/bin/python
#-*- coding:utF-8 -*-

import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
import numpy as np

# 设置缺失值类型
na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']

# 打开当前目录下的数据文件
pred_df = pd.read_table('./data_set_score.txt',sep = '\t',na_values = na_values)

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


# a,b值用于后续修正score评分
print a,b







