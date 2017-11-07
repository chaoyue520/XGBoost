#!/usr/bin/python
#-*- coding:utF-8 -*-

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
import os
import sys
import gc
import re


# 查看路径
os.getcwd()
# 变更路径
os.chdir('')   

### 加载数据 #####
######## 1、异常值统一替换标准
######## 2、读取列名var_name_v1=var_name['var_name'].values
######## 3、加载数据，并对数据列名重新赋值。注意:对于nan值不再做统一替换，容易报内存不足。data_set_v0.fillna('-99999') 
######## 4、概览数据
######## 5、如有需要，可调整部分字段数据类型

xpath='/home/fsg/jiwenchao/ac_class/data'

#1、设置异常值替换标准
na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']

xfile_name='/data_set_name.txt'
#2、读取列名
data_set_name=pd.read_table(xpath+xfile_name,
                         header=None,
                         names=['var_name']
                         )

xfile_data='/data_set.txt'
#3、加载数据
data_set_v0=pd.read_table(xpath+xfile_data,
                         sep='\t',
                         header=None,
                         na_values=na_values,
                         names=data_set_name['var_name'].values
                         )

#4、可以考虑将所有nan值统一替换为-999，而不用想fillna函数那样需要重新赋值，节约内存空间
#data_set_v0.replace(to_replace=na_values, value='-9999', inplace=True)

#4、概览数据
def summ_col_x(df):
     a=df.shape
     b=df.groupby(['risk_tag_union'])['passid'].count()
     c=df.dtypes
     return c,a,b

#5、调整数据类型
#data_set_v0['risk_tag_union']=data_set_v0['risk_tag_union'].astype(object)

### 首先对特殊变量重新赋值 #####
######## 1、对card_type变量重新赋值，因为card_type变量除了正常数值以外，还有部分数值是异常值，对异常值做统一替换
######## 注意：先不要直接使用drop函数，将需要删除的字段统一放在一起，最终一并删除

#1、card_type重新赋值
try:
     data_set_v0.loc[data_set_v0.card_type == '储蓄卡', 'card_type_new'] = 'card_chu' 
     data_set_v0.loc[data_set_v0.card_type == '信用卡', 'card_type_new'] = 'card_xin' 
     data_set_v0.loc[(data_set_v0.card_type != '储蓄卡') & (data_set_v0.card_type != '信用卡'), 'card_type_new'] = 'card_other'
     #data_set_v0=data_set_v0.drop(['card_type'],axis=1)   #对原始字段转换后，随即删除该字段，保留匹配后的字段
except exception,e:
     print e.message
     print data_set_v0.groupby(['card_type'])['risk_tag_union'].sum()
     print data_set_v0.groupby(['card_type_new'])['risk_tag_union'].sum()
     #print data_set_v0[['age','card_type', 'tip']] 

#data_set_v0.shape
#data_set_v0['card_type_new'].value_counts()

# 对 reg_city 按照城市等级进行分类 ###
######## 1、加载城市码表，码表按照中国城市等级划分，从一线城市到五线城市。也可以考虑通过城市聚类实现
######## 2、去掉原始数据中城市字段的 '市'
######## 3、left join 匹配附表数据
######## 4、从关联后的结果中删掉关联字段，如reg_city，pay_city

city_index='/city_index.txt'

#1、加载城市码表
city_index = pd.read_table(xpath+city_index, sep = '\t')

#2、删除reg_city字段中的 '市' 字段，替换为空值
data_set_v0.reg_city.replace(to_replace = '市', value = '', inplace = True, regex = True)

#3.1、注册城市
reg_city_idx = city_index.rename_axis({'city': 'reg_city', 'index': 'reg_city_idx'}, axis='columns')
data_set_v0 = pd.merge(data_set_v0, reg_city_idx, on = 'reg_city', how = 'left')
#data_set_v0.shape

#3.2、支付城市
#pay_city_idx = city_index.rename_axis({'city': 'pay_city', 'index': 'pay_city_idx'}, axis='columns')
#data_set_v0 = pd.merge(data_set_v0, pay_city_idx, on = 'pay_city', how = 'left')

#4、删除原始reg_city字段，只保留city_index字段  data_set_v0 = data_set_v0.drop(['reg_city','pay_city'], axis=1)
#注意：此处需要变更 reg_city_idx 字段属性为分类型
data_set_v0.groupby(['reg_city_idx'])['risk_tag_union'].sum()

data_set_v0['reg_city_idx']=data_set_v0['reg_city_idx'].astype(object)


### 筛选出nuLL值比例超过阈值的列#####
######## 1、挑选出nuLL值比例超过阈值的列
######## 2、计算被挑选的列对标签risk_tag的覆盖率，挽回覆盖率超过阈值的列
######## 3、结合1和2筛选出最终需要剔除的列

#1、筛选出nuLL值比例超过0.95的列，手动判断被删除变量的区分度
def excld_na_cols(df,p):
     excld_na_list=[]
     for col in df.columns:
          if ((np.sum(pd.isnull(df[col]))+np.sum(df[col]<0))*1.0/df.shape[0])>p:
               excld_na_list.append(col)
     #df = df.drop(excl_list,axis=1)
     return excld_na_list

#筛选出NULL比例超过95%
excld_na_list=excld_na_cols(data_set_v0,0.95)


print gc.collect()
#data_set_v0['b_sweep_c_type'].value_counts()
#data_set_v0.groupby('derive_frontblack_flag_pay_last_1day')['risk_tag_union'].sum()


#2、对于已删除的nuLL值比例超过0.95的列，重新计算非nuLL值对risk_tag值的覆盖，如果覆盖的比例超过p，则重新拉回
# 注意：f1_need_save_cols 计算过于耗内存，无法大量计算；该方法等价于need_save_cols
def f1_need_save_cols(df,p):
     X_col_risktag=[0 for col in range(df.shape[0])] 
     save_cols_list=[]
     for col in excld_na_list:
          for i in range(df.shape[0]):
               if pd.notnull(df[col])[i]==True:
                    X_col_risktag[i]=df['risk_tag_union'][i]
               elif pd.notnull(df[col])[i]==False:
                    X_col_risktag[i]=0
          if sum(X_col_risktag)*1.0/df['risk_tag_union'].sum() >= p:
               save_cols_list.append(col)
     return save_cols_list


def need_save_cols(df,p):
     X_col_risktag=[]
     save_cols_list=[]
     for col in excld_na_list:
          X_col_risktag.append(sum(df.groupby([col])['risk_tag_union'].sum()))
     for i in range(len(X_col_risktag)):
          if X_col_risktag[i]*1.0/df['risk_tag_union'].sum()>p:
               save_cols_list.append(excld_na_list[i])
     return save_cols_list

#挽回比例阈值0.02
save_cols_list=need_save_cols(data_set_v0,0.02)

#判断标签是否在删除的变量列表中
delete_na_list=[]
for col in excld_na_list:
     if col not in save_cols_list:
          delete_na_list.append(col)


'risk_tag_union' in delete_na_list


### 列包含数值数量判断 #####
######## 1、特殊地，首先筛选唯一值的列，nan也计算在内，即如果一列只包含1和nan，则该字段包含数值个数为2，而不是1，因为nan值也计算在内了
######## 2、列出所有包含唯一值的列，对于只含有唯一值的列，处理办法是直接删除
######## 3、筛选离散和连续变量值比较的多的列，后续可做分箱处理

#1、计算含有唯一值的列
def excld_unique_col(df):
     excld_unique_list = []
     for col in df.columns:
          if len(df[col].value_counts(dropna=False)) == 1 : # NaN值也计算在内，value_counts()默认是不计算nan的数量，可以通过dropna设置
               excld_unique_list.append(col)
     return excld_unique_list


#2、列出所有包含唯一值的列
delete_unique_list=excld_unique_col(data_set_v0)


#3、离散和连续变量值比较的多的列
def get_more_vlaue_col(df):
     num_more_vlaue_list = []
     non_num_more_vlaue_list=[]
     for col in df.columns:
          if df[col].dtype != 'object' and len(df[col].value_counts(dropna=True)) >= 1000 :
               num_more_vlaue_list.append(col)
          elif df[col].dtype == 'object' and len(df[col].value_counts(dropna=True)) >= 10 :
               non_num_more_vlaue_list.append(col)
     return num_more_vlaue_list,non_num_more_vlaue_list

#判断分类变量多的列的值分布，考虑做处理
mutil_value_col_list=get_more_vlaue_col(data_set_v0)


####分类变量做处理：zhu_degree

#NULL值判断
sum(data_set_v0['zhu_degree'].isnull())
#标签覆盖
data_set_v0.groupby('zhu_degree')['risk_tag_union'].sum()
#特殊值判断
data_set_v0[data_set_v0.zhu_degree=='4']['zhu_degree'].count()
#多分类合并
try:
     data_set_v0.loc[(data_set_v0.zhu_degree == '99其他') | (data_set_v0.zhu_degree == '其他'), 'zhu_degree_new'] = 'zhu_degree_other' 
     data_set_v0.loc[(data_set_v0.zhu_degree == '硕士研究生') | (data_set_v0.zhu_degree == '博士研究生') | (data_set_v0.zhu_degree == '03硕士研究生') | (data_set_v0.zhu_degree == '04博士研究生'), 'zhu_degree_new'] = 'zhu_degree_shuo' 
     data_set_v0.loc[(data_set_v0.zhu_degree == '本科') | (data_set_v0.zhu_degree == '02本科') | (data_set_v0.zhu_degree == '专升本') | (data_set_v0.zhu_degree == '第二学士学位'), 'zhu_degree_new'] = 'zhu_degree_ben' 
     data_set_v0.loc[(data_set_v0.zhu_degree == '01专科') | (data_set_v0.zhu_degree == '专科') | (data_set_v0.zhu_degree == '专科(高职)') | (data_set_v0.zhu_degree == '夜大电大函大普通班'), 'zhu_degree_new'] = 'zhu_degree_zhuan'
     data_set_v0.loc[(data_set_v0.zhu_degree == '4') | (data_set_v0.zhu_degree == '5') | (data_set_v0.zhu_degree == '6') | (data_set_v0.zhu_degree == '7'), 'zhu_degree_new'] = 'zhu_degree_other'
     #data_set_v0=data_set_v0.drop(['zhu_degree'],axis=1)   #对原始字段转换后，随即删除该字段，保留匹配后的字段
except exception,e:
     print e.message
     print data_set_v0.groupby(['zhu_degree_new'])['risk_tag_union'].sum()
     #print data_set_v0[['age','zhu_degree', 'tip']] 

#判断结果
data_set_v0['zhu_degree_new'].value_counts()
sum(data_set_v0['zhu_degree_new'].isnull())
### 极端值判断和替换步骤 #####
######## 1、计算是否有极值,判断依据：如果该列的最大值大于99分位数的q倍，标记该列为包含极端值的列
######## 2、确定筛选标准以及确定最终需要做极端值处理的列，
######## 3、注意，经过get_extreme_col作用后，df的极端值已经被替换掉，只是最终extreme_list，而不打印出来df

#1、挑选含有极端值的列
def judge_extreme_col(df,p,threshold):
     extreme_list=[]
     for col in df.columns:
          if df[col].dtype != 'object' and df[col].max()>=50000 and df[col].quantile(threshold)>0 and df[col].max() > p*df[col].quantile(threshold):
               extreme_list.append(col)
               #df.loc[df[col]>df[col].quantile(q=threshold),col]=df[col].quantile(q=threshold)   #需要修改 inf>10 True
     return extreme_list


# 1000倍于99.95分位数，则判断为疑似存在极端值的列
get_extreme_col_list=judge_extreme_col(data_set_v0,1000,0.9995)


#2、确定筛选标准以及确定最终需要做极端值处理的列,并describe需要处理的极端值的列，作为第三步操作的对比
def get_extreme_col(df):
     extreme_list=get_extreme_col_list   # p,threshold跟judge_extreme_col参数保持一致
     Recover_cols_v1=['tieba_post_yule_rcy3m','tieba_post_yule_rcy6m','tieba_post_yule_rcyth1m','tieba_view_gongnongye_rcy3m','tieba_view_gongnongye_rcy6m','tieba_view_gongnongye_rcyth1m']  #手动筛选不需要做极值处理的字段
     extreme_list_desc=[]
     extreme_list_cols=[]
     for col in extreme_list:
          if col not in Recover_cols_v1:
               extreme_list_desc.append(df[col].describe())
               extreme_list_cols.append(col)
     return extreme_list_desc,extreme_list_cols

#3、对最终挑选的极端值列做分位数替换处理，直接替换
def transfer_extreme_col(df,threshold):
     transfer_extreme_Desc=[]
     transfer_extreme_cols=[]
     Recover_cols_v1=['tieba_post_yule_rcy3m','tieba_post_yule_rcy6m','tieba_post_yule_rcyth1m','tieba_view_gongnongye_rcy3m','tieba_view_gongnongye_rcy6m','tieba_view_gongnongye_rcyth1m']  #手动筛选不需要做极值处理的字段
     for col in get_extreme_col_list:
          if col not in Recover_cols_v1:
               df.loc[df[col]>df[col].quantile(q=threshold),col]=df[col].quantile(q=threshold)
               transfer_extreme_Desc.append(df[col].describe())
               transfer_extreme_cols.append(col)
     return transfer_extreme_Desc,transfer_extreme_cols


transfer_extreme_col_list=transfer_extreme_col(data_set_v0,0.9995)

### 连续变量分箱步骤 #########
######## 1、筛选出需要分享的连续变量：need_binning_cols  &  X_col in need_binning_cols
######## 2、计算变量在不同分箱下的iv值，确定每个需要分箱变量最终的分箱数
######## 3、将已分箱的变量按照分箱重新赋值
######## 4、删除分箱的变量，保留分箱变量变换后的变量，统一加上 _batch 后缀 
######## 5、通过生成的新变量 *_batch 生成新的数据集 data_set_v1

#1、筛选需要分箱的连续变量,原则筛选变量数多于100个，且存在极值的变量, q>1 , 0 < threshold < 1  ,  p>=1
def ex_more_vlaue_col(df,q,threshold,p):
     ex_more_vlaue_col = []
     for col in df.columns:
          if df[col].dtype != 'object' and len(df[col].value_counts()) >= q and df[col].max()/df[col].quantile(threshold) > p:
               ex_more_vlaue_col.append(col)
     return ex_more_vlaue_col
 

#need_binning_cols=ex_more_vlaue_col(df,1000,0.9995,1000)

need_binning_cols=get_extreme_col_list

#data_set_v0.groupby(['payamt_10_times'])['risk_tag_union'].sum()[0:40]

############# times ###########
#2.1、通过手动设置cutpoints计算IV值
def calciv_set_cutpoints(X_col,Y_col):
     a=[0,1,2,3,4,5,6,7,8,11,20,30,100]
     woe=np.zeros(np.unique(a).shape)
     n_0 = np.sum(Y_col==0)
     n_1 = np.sum(Y_col==1)
     n_0_group = np.zeros(np.unique(a).shape)
     n_1_group = np.zeros(np.unique(a).shape)
     np.unique(a).sort()
     for i in range(len(np.unique(a))):
          if i < max(range(len(np.unique(a))-1)):
               n_0_group[i] = Y_col[(X_col >= a[i]) & (X_col < a[i+1]) & (Y_col == 0)].count()
               n_1_group[i] = Y_col[(X_col >= a[i]) & (X_col < a[i+1]) & (Y_col == 1)].count()
               woe[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
          elif i==len(np.unique(a))-1:
               n_0_group[i] = Y_col[(X_col >= a[i]) & (Y_col == 0)].count()
               n_1_group[i] = Y_col[(X_col >= a[i]) & (Y_col == 1)].count()
               woe[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
     iv=sum(woe)
     return  iv,woe

#批量筛选需要手动设置cutpoint的变量
need_set_cutpoints_col_list=[]
for col in get_extreme_col_list:
     if col.endswith('times'):
          need_set_cutpoints_col_list.append(col)

Y_col=data_set_v0['risk_tag_union']
#批量计算iv值
cutpoint_iv_value=[]
for col in need_set_cutpoints_col_list:
     X_col=data_set_v0[col]
     cutpoint_iv_value.append(calciv_set_cutpoints(X_col,Y_col)[0])

cutpoint_iv_value

#3.1、数据分箱后批量替换原始变量
for col in need_set_cutpoints_col_list:
     a=[0,1,2,3,4,5,6,7,8,11,20,30,100]
     for i in range(len(np.unique(a))):
          if i <= max(range(len(np.unique(a))-1)):
               data_set_v0.loc[(data_set_v0[col]>= a[i]) & (data_set_v0[col]<=a[i+1]),col+'_batch']=col+'_'+str(i)
          elif i==max(range(len(np.unique(a)))):
               data_set_v0.loc[data_set_v0[col]>=a[i],col+'_batch']=col+'_'+str(i)

#分类后的分布
data_set_v0[['payamt_10_times','payamt_10_times_batch']].iloc[0:10,0:2]
data_set_v0['payamt_10_times'].value_counts()
data_set_v0['payamt_10_times_batch'].value_counts()
data_set_v0.groupby(['payamt_10_times_batch'])['risk_tag_union'].sum()
############# stddev #############
#2.1、通过手动设置cutpoints计算IV值
def calciv_equal_percent(X_col,Y_col,k,point):
     a=[0 for i in range(k)]
     for i in range(k):
          a[i]=stats.scoreatpercentile(X_col, i*point) # point <= 100/(k-1) 
     woe=np.zeros(np.unique(a).shape)
     n_0 = np.sum(Y_col==0)
     n_1 = np.sum(Y_col==1)
     n_0_group = np.zeros(np.unique(a).shape)
     n_1_group = np.zeros(np.unique(a).shape)
     for i in range(len(np.unique(a))):
          if i < max(range(len(np.unique(a))-1)):
               n_0_group[i] = Y_col[(X_col >= a[i]) & (X_col < a[i+1]) & (Y_col == 0)].count()
               n_1_group[i] = Y_col[(X_col >= a[i]) & (X_col < a[i+1]) & (Y_col == 1)].count()
               woe[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
          elif i==len(np.unique(a))-1:
               n_0_group[i] = Y_col[(X_col >= a[i]) & (Y_col == 0)].count()
               n_1_group[i] = Y_col[(X_col >= a[i]) & (Y_col == 1)].count()
               woe[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
     iv=sum(woe)
     return  iv,woe

#批量筛选需要手动设置cutpoint的变量
need_quantile_col_list_v1=[]
for col in get_extreme_col_list:
     if col.startswith('stddev'):
          need_quantile_col_list_v1.append(col)

#批量计算iv值，k=5,point=10
quantile_iv_value=[]
for col in need_quantile_col_list_v1:
     X_col=data_set_v0[col]
     quantile_iv_value.append(calciv_equal_percent(X_col,Y_col,5,10)[0])  

#批量筛选需要手动设置cutpoint的变量
need_quantile_col_list_v1=['stddev_payamt_uid_15_90','stddev_payamt_uid_15_120','stddev_payamt_uid_15_180','stddev_payamt_uid_15_240']

#3.1、数据分箱后批量替换原始变量
a=[0]*5   # k=5
for col in need_quantile_col_list_v1:
     for i in range(5):   # k=5
          a[i]=stats.scoreatpercentile(data_set_v0[col], 10*i)  # point =10
     for i in range(len(np.unique(a))):
          if i <= max(range(len(np.unique(a))-1)):
               data_set_v0.loc[(data_set_v0[col]>= a[i]) & (data_set_v0[col]<=a[i+1]),col+'_batch']=col+'_'+str(i)
          elif i==max(range(len(np.unique(a)))):
               data_set_v0.loc[data_set_v0[col]>=a[i],col+'_batch']=col+'_'+str(i)

#分类后的分布
data_set_v0[['stddev_payamt_uid_15_90','stddev_payamt_uid_15_90_batch']].iloc[0:10,0:2]
data_set_v0['stddev_payamt_uid_15_90'].value_counts()
data_set_v0['stddev_payamt_uid_15_90_batch'].value_counts()
data_set_v0.groupby(['stddev_payamt_uid_15_90_batch'])['risk_tag_union'].sum()

########### sum_payamt ##########
#2.1、通过手动设置cutpoints计算IV值
def calciv_equal_percent(X_col,Y_col,k,point):
     a=[0 for i in range(k)]
     for i in range(k):
          a[i]=stats.scoreatpercentile(X_col, i*point) # point <= 100/(k-1) 
     woe=np.zeros(np.unique(a).shape)
     n_0 = np.sum(Y_col==0)
     n_1 = np.sum(Y_col==1)
     n_0_group = np.zeros(np.unique(a).shape)
     n_1_group = np.zeros(np.unique(a).shape)
     for i in range(len(np.unique(a))):
          if i < max(range(len(np.unique(a))-1)):
               n_0_group[i] = Y_col[(X_col >= a[i]) & (X_col < a[i+1]) & (Y_col == 0)].count()
               n_1_group[i] = Y_col[(X_col >= a[i]) & (X_col < a[i+1]) & (Y_col == 1)].count()
               woe[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
          elif i==len(np.unique(a))-1:
               n_0_group[i] = Y_col[(X_col >= a[i]) & (Y_col == 0)].count()
               n_1_group[i] = Y_col[(X_col >= a[i]) & (Y_col == 1)].count()
               woe[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
     iv=sum(woe)
     return  iv,woe


#批量筛选需要batch的变量正则筛选需要的列                                                                           ###
need_quantile_col_list_v2=[]
for col in get_extreme_col_list:
     patt=r"(sum)"
     patt1=r"(state1)"
     m=re.search(patt,col)
     m1=re.search(patt1,col)
     if m is not None and m1 is None:
          need_quantile_col_list_v2.append(col)
          print col , ': match '
     else:
          print col , ': not match '

#批量计算iv值
quantile_iv_value=[]
for col in need_quantile_col_list_v2:
     X_col=data_set_v0[col]
     quantile_iv_value.append(calciv_equal_percent(X_col,Y_col,10,5)[0])

quantile_iv_value

#剔除IV值为NaN值的变量
need_quantile_col_list_v3=list(set(need_quantile_col_list_v2
     +['payamount','quzheng_payamt','device_base_payamt','device_pre_payamt','device_pre_quzheng_payamt','uid_base_payamt','uid_pre_payamt','uid_pre_quzheng_payamt'])
     -set(['uid_sum_payamt_last_30min', 'uid_sum_payamt_reject_last_30min', 'uid_sum_payamt_last_1day'])
     -set(['device_sum_payamt_reject_last_30min', 'device_sum_payamt_last_1day', 'device_sum_payamt_last_30min'])
     )

a=[0]*10   # k=5
#3.1、数据分箱后批量替换原始变量
for col in need_quantile_col_list_v3:
     for i in range(10):   # k=5
          a[i]=stats.scoreatpercentile(data_set_v0[col], 5*i)  # point =10
     for i in range(len(np.unique(a))):
          if i <= max(range(len(np.unique(a))-1)):
               data_set_v0.loc[(data_set_v0[col]>= a[i]) & (data_set_v0[col]<=a[i+1]),col+'_batch']=col+'_'+str(i)
          elif i==max(range(len(np.unique(a)))):
               data_set_v0.loc[data_set_v0[col]>=a[i],col+'_batch']=col+'_'+str(i)

#分类后的分布
data_set_v0[['payamount','payamount_batch']].iloc[0:10,0:2]
data_set_v0['payamount'].value_counts()
data_set_v0['payamount_batch'].value_counts()
data_set_v0.groupby(['payamount_batch'])['risk_tag_union'].sum()

########### 如果不分箱 ################
#2.3、如果不分箱iv值
def calciv(X_col,Y_col):
     iv=np.zeros(np.unique(X_col).shape)
     n_0  = np.sum(Y_col==0)
     n_1 = np.sum(Y_col==1)
     n_0_group = np.zeros(np.unique(X_col).shape)
     n_1_group = np.zeros(np.unique(X_col).shape)
     np.unique(X_col).sort()
     for i in range(len(np.unique(X_col))):
          n_0_group[i] = Y_col[(X_col == np.unique(X_col)[i]) & (Y_col == 0)].count()
          n_1_group[i] = Y_col[(X_col == np.unique(X_col)[i]) & (Y_col == 1)].count()
          iv[i] = (np.sum(n_0_group[i])/n_0-np.sum(n_1_group[i])/n_1)*np.log(((np.sum(n_0_group[i])/n_0)/(np.sum(n_1_group[i])/n_1)))
     iv=sum(iv)
     return  iv


# 所有分箱变量
need_binning_cols=need_set_cutpoints_col_list+need_quantile_col_list_v1+need_quantile_col_list_v3


### 筛选无效变量 ###############
######## 1、首先提出无效变量，如dt等时间变量，cuid等匹配关联变量
######## 注意:至少保留passid字段和sessionid字段，以备后续模型预测值可以有效定位到某个用户

#1、踢掉无效变量
def excld_match_col(df):
     excld_invalid=[]
     for col in df.columns:
          if col.startswith('dt') or col.startswith('data_set_seg')  or col.startswith('userid') or col.startswith('device_cookie_id') or col.startswith('payeebankcard_1')or col.startswith('osname') or col.startswith('dasou_dt_month') or col.startswith('uid_base_pre_sessionid') or col.startswith('reg_country') or col.startswith('dasou_passid'):
            #or col.startswith('sessionid') or col.startswith('userid') or col.startswith('passid'):
               excld_invalid.append(col)
     excld_invalid=excld_invalid+['risk_tag_both']+['card_type']+['reg_city']+['zhu_degree']
     excld_invalid_list=list(df.columns.intersection(excld_invalid))  #交集
     #df1=df.drop(custom_excl_list,axis=1)
     return excld_invalid_list


########### 所有需要删除变量汇总 ######################
######## 1、无效变量列表
######## 2、NULL值比例超过95%，且对Y标签的覆盖低于2%
######## 3、删除唯一值列

#1、无效变量
excld_invalid_list = excld_match_col(data_set_v0)

#2、NULL值
delete_na_list=[]
for col in excld_na_list:
     if col not in save_cols_list:
          delete_na_list.append(col)

delete_na_list

#3、唯一值列
delete_unique_list=excld_unique_col(data_set_v0)

# 总结：所有需要删除的变量
#注意看明细：all_need_delete_cols，避免误删变量
all_need_delete_cols=excld_invalid_list+delete_na_list+delete_unique_list


########### 剔除无效变量，生成新的数据集 ######################
####### 注意：valid_cols=list(all_col.difference(all_excld_cols))


data_var_names_arr=data_set_v0.columns.values
var_names=list(set(data_var_names_arr)-set(all_need_delete_cols))  #set()特点：无序不重复集合

data_set_v1=data_set_v0[var_names]

print data_set_v0.shape
print data_set_v1.shape

#补充说明：释放内存
del data_set_v0
print gc.collect()

########分类变量挑选######
######## 1、挑选分类变量列，后续需要离散化
######## 2、将数据分成离散列和连续列
######## 3、将分类变量离散化，离散化的过程中，对于nan是不做处理的，即只离散化分类变量的数值型value，而对nan值不做处理
######## 4、合并先前的连续变量和后续离散化的分类变量，并生成新的数据集
######## 注意：逻辑判断，变量筛选过程中，至少要保留passid等用户身份信息，方便在模型预测阶段定位信息

#1、筛选所有离散变量
def get_obj_cols(df):
     obj_col_list=[]
     for col in df.columns:
          if df[col].dtype == 'object':
               obj_col_list.append(col)
     return obj_col_list

# 注意:此处最好能改看下 obj_col_list 具体包含哪些变量
obj_col_list=get_obj_cols(data_set_v1)

#sessionid也被划分为分类变量，需要剔除,剔除sessionid后，分类变量在data_set_num数据集中
obj_col_list.remove('sessionid')

#2、筛选出分类变量和连续变量，以及sessionid
data_set_obj=data_set_v1[obj_col_list]
data_set_num=data_set_v1.drop(obj_col_list,axis=1)


#3、分类变量离散化
data_set_obj_trans_num=pd.get_dummies(data = data_set_obj,prefix=data_set_obj.columns)

#补充说明：释放内存
del data_set_v1
print gc.collect()

#4、合并离散化后的分类变量和连续变量
data_set=pd.concat([data_set_obj_trans_num,data_set_num],axis=1)

data_set.shape

#5、逻辑判断
'data_set_tag' in data_set.columns
'risk_tag_union' in data_set.columns
'passid' in data_set.columns
'sessionid' in data_set.columns
'card_type_new' in data_set.columns
'zhu_degree_new' in data_set.columns


data_set.groupby(['risk_tag_union'])['passid'].count()

######## 数据预处理完毕，保存新的数据集######
######## 1、导出文件
######## 2、替换所有nan值为空值，视情况而定


#1、导出文件
data_set.to_csv(xpath+'/data_set_0705.txt',sep='\t',header=True,index=False)



