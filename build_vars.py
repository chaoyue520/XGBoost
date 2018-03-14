#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import pandas as pd
import math

reload(sys)
sys.setdefaultencoding('utf-8')

# argv ------------------------------------------------------------

cur_mon = sys.argv[1]
last_mon = sys.argv[2]
n_cache = 10000	# 批处理样本数

# na_values_li ----------------------------------------------------

na_values_li = ['NA','na','NaN','nan','\\N','NULL','null','']

# 大数据变量名 ----------------------------------------------------

bd_var_name_file = './data/credit_fraud_var_name.txt'
bd_var_names_arr = pd.read_table(bd_var_name_file,
								 sep = '\t',
								 header = None,
								 names = ['var_name']).var_name.values

# 上月变量的名字加后缀 '_0'
bd_var_names_arr_lm = []
for i in range(len(bd_var_names_arr)):
	if bd_var_names_arr[i] != 'passid':
		bd_var_names_arr_lm.append(str(bd_var_names_arr[i]) + '_0')
	else:
		bd_var_names_arr_lm.append(str(bd_var_names_arr[i]))

bd_var_cate = ['educational_level', 't3', 'iedu', 'job', 'tc1', 'tc2', 'tc3', 'tc4', 
			   'tc5', 'tc6', 'company_city', 'home_city', 'local_city', 'card_type',
			   'reg_city', 'reg_prove', 'first_level_richness', 'erised_applist',
 			   'erised_taglist', 'erised_user_poi',
 			   'educational_level_0', 't3_0', 'iedu_0', 'job_0', 'tc1_0', 'tc2_0', 
 			   'tc3_0', 'tc4_0', 'tc5_0', 'tc6_0', 'company_city_0', 'home_city_0', 
 			   'local_city_0', 'card_type_0', 'reg_city_0', 'reg_prove_0', 
 			   'first_level_richness_0', 'erised_applist_0', 'erised_taglist_0', 
			   'erised_user_poi_0']
		
# 离线表 varname --------------------------------------------------

offline_table_varname_file = './data/offline_table_varname.txt'
tbl_varname_arr = pd.read_table(offline_table_varname_file,
							    sep = '\t',
							    header = None,
							    names = ['var_name']).var_name.values

# 码表 ------------------------------------------------------------

# 城市等级
city_idx_map_file = './data/city_index.txt'
city_index = pd.read_table(city_idx_map_file, sep = '\t')

# 搜索词聚类字典
cluster_class_file = './data/search_word_classes.sorted.txt'
n_class = 500
word_dict_cls = {}
for each in open(cluster_class_file, 'r'):
    word_class = str(each).strip().split(' ')
    word_dict_cls[word_class[0]] = int(word_class[1])

# 风险搜索词库
risk_word = pd.read_table('./data/risk_search_word_set.txt',
						  sep = '\t',
						  header = None,
						  names = ['var_name']).var_name.values
risk_word_var = ['search_word_' + risk_word[i] for i in range(len(risk_word))]

# app分类字典
app_class_file = './data/app_class_by_label.txt'
app_dict_cls = {}
app_class_set = set()
for each in open(app_class_file, 'r'):
    app_class = str(each).strip().split('\t')
    if len(app_class) > 1:
        app_dict_cls[app_class[0]] = app_class[1]
        app_class_set.add(app_class[1])
app_classes = list(app_class_set)
n_app_class = len(app_classes)

# 风险app库
risk_app_file = './data/risk_app_set.txt'
risk_app = pd.read_table(risk_app_file,
						 sep = '\t',
						 header = None,
						 names = ['var_name']).var_name.values
risk_app_var = ['risk_app_' + risk_app[i] for i in range(len(risk_app))]

# Good LBS 库
good_lbs = pd.read_table('./data/good_lbs.txt',
						 sep = '\t',
						 header = None,
						 names = ['var_name']).var_name.values
good_lbs_var = ['lbs_' + good_lbs[i] for i in range(len(good_lbs))]
	
# Top 6 兴趣
bad_intes = set(['贷款','银行业务','信用卡','金融财经','软件应用','网游'
				,'持续在线','酒水','星座运势','房屋租赁','酒店星级/商务'
				,'征信','非汽车类机动车','整容整形','博彩彩票','间断在线'
				])
good_intes = set(['旅游出行','医疗健康','汽车','家电数码','个用家电','房产'
				 ,'教育培训','母婴亲子','个护美容','基础教育','技能培训'
				 ,'出国游','建材家居','港澳台','家用家电','保险','飞机'
				 ,'新能源汽车','旅游出行/飞机','东南亚','股票','家教','创业'
				 ])

# Function --------------------------------------------------------

def index(li, item):
	# 返回item在列表li中第一次出现的index，未找到返回-1
	if len(li) > 0 and item != None:
		for i in range(len(li)):
			if li[i] == item: return i
	return -1

		
def build_var(data_df):

	# 确定变量类型 ------------------------------------------------
	data_df.replace(to_replace=na_values_li, value=np.nan, inplace=True)
	cols = list(set(data_df.columns.values) - set(['passid']))
	for i in range(len(cols)):
		try: data_df[cols[i]] = data_df[cols[i]].astype(float)
		except:	pass
	for i in range(len(bd_var_cate)):
		data_df[bd_var_cate[i]] = data_df[bd_var_cate[i]].astype(object)
	
	# 画像数据衍生 ------------------------------------------------
	
	# 求画像变量的环比：[类别型变量：与上月是否相同；数值型变量：较上月的增长率]

	var_names = list(set(bd_var_names_arr) - set(['passid', 'erised_applist', 'erised_taglist', 'erised_user_poi']))
	var_names_0 = []
	for i in range(len(var_names)):
		var_names_0.append(str(var_names[i]) + '_0')

	for i in range(len(var_names)):
		if data_df[var_names[i]].dtype != 'object' and data_df[var_names_0[i]].dtype != 'object':
			data_df[str(var_names[i]) + '_rate'] = list((data_df[var_names[i]] - data_df[var_names_0[i]]) / (data_df[var_names_0[i]] + 0.001))
		else:
			data_df[str(var_names[i]) + '_iseq'] = list((data_df[var_names[i]] == data_df[var_names_0[i]]).astype(float))

	rm_vars = ['erised_applist_0', 'erised_taglist_0', 'erised_user_poi_0']
	data_df = data_df.drop(rm_vars, axis = 1)
	
	# 信息厚度 ----------------------------------------------------

	rich_vars = ['first_level_richness', 'first_level_richness_0']
	rich_var_new = ['bd_richness', 'bd_richness_0']
	rich_vars_dtl = []
	# 本月
	for i in range(35):
		rich_vars_dtl.append('bd_richness_' + str(i+1))
	rich_vars_dtl_0 = []
	# 上月
	for i in range(35):
		rich_vars_dtl_0.append('bd_richness_0_' + str(i+1))

	# 信息厚度取值

	for i in range(len(rich_vars)):
		bd_richness = data_df[rich_vars[i]].str.slice(0, 2)
		bd_richness[bd_richness.eq('')] = np.nan
		bd_richness.replace(to_replace='_', value='', inplace=True, regex=True)
		bd_richness = bd_richness.astype(np.float)
		data_df[rich_var_new[i]] = bd_richness
	bd_richness = None

	# 每一维主key的取值
	# 本月
	bd_richness_dtl = data_df[rich_vars[0]].str.slice(start=2)
	bd_richness_dtl[bd_richness_dtl.eq('')] = np.nan 
	bd_richness_dtl.replace(to_replace='_', value='', inplace=True, regex=True)
	bd_richness_dtl = bd_richness_dtl.astype(object)
	for j in range(len(rich_vars_dtl)):
		data_df[rich_vars_dtl[j]] = bd_richness_dtl.str.slice(j, j+1).astype(np.float)

	# 上月
	bd_richness_dtl = data_df[rich_vars[1]].str.slice(start=2)
	bd_richness_dtl[bd_richness_dtl.eq('')] = np.nan 
	bd_richness_dtl.replace(to_replace='_', value='', inplace=True, regex=True)
	bd_richness_dtl = bd_richness_dtl.astype(object)
	for j in range(len(rich_vars_dtl_0)):
		data_df[rich_vars_dtl_0[j]] = bd_richness_dtl.str.slice(j, j+1).astype(np.float)
	bd_richness_dtl = None

	# 对比上下两个月

	data_df['bd_richness_iseq'] = list((data_df['bd_richness'] == data_df['bd_richness_0']).astype(float))
	for i in range(len(rich_vars_dtl)):
		data_df[str(rich_vars_dtl[i]) + '_iseq'] = list((data_df[rich_vars_dtl[i]] == data_df[rich_vars_dtl_0[i]]).astype(float))

	data_df = data_df.drop(rich_vars, axis = 1)

	# 城市等级 -----------------------------------------------------

	# format city 
	data_df.home_city.replace(to_replace = '市', value = '', inplace = True, regex = True)
	data_df.company_city.replace(to_replace = '市', value = '', inplace = True, regex = True)
	data_df.local_city.replace(to_replace = '市', value = '', inplace = True, regex = True)
	data_df.reg_city.replace(to_replace = '市', value = '', inplace = True, regex = True)
	
	# 家庭所在城市
	home_city_idx = city_index.rename_axis({'city': 'home_city', 'index': 'home_city_idx'}, axis='columns')
	data_df = pd.merge(data_df, home_city_idx, on = 'home_city', how = 'left')

	# 工作所在城市
	comp_city_idx = city_index.rename_axis({'city': 'company_city', 'index': 'company_city_idx'}, axis='columns')
	data_df = pd.merge(data_df, comp_city_idx, on = 'company_city', how = 'left')
	
	# 本地常访地所在城市
	local_city_idx = city_index.rename_axis({'city': 'local_city', 'index': 'local_city_idx'}, axis='columns')
	data_df = pd.merge(data_df, local_city_idx, on = 'local_city', how = 'left')

	# 注册城市
	reg_city_idx = city_index.rename_axis({'city': 'reg_city', 'index': 'reg_city_idx'}, axis='columns')
	data_df = pd.merge(data_df, reg_city_idx, on = 'reg_city', how = 'left')

	data_df = data_df.drop(['home_city', 'company_city', 'local_city', 'reg_city',
							'home_city_0', 'company_city_0', 'local_city_0', 'reg_city_0'], axis=1)

	# 教育水平 ----------------------------------------------------

	try:
		data_df.loc[data_df.educational_level == '高中及以下', 'educational_lvl'] = 1 
		data_df.loc[data_df.educational_level == '大专', 'educational_lvl'] = 2 
		data_df.loc[data_df.educational_level == '本科及以上', 'educational_lvl'] = 3
		data_df = data_df.drop(['educational_level'], axis=1)
	except:
		print(data_df[['passid', 'educational_level', 'educational_level_0']])

	# 搜索词 word2vec聚类统计 -------------------------------------
	
	# 统计用户每类搜索词的频次
	search_words = list(data_df.erised_taglist)
	class_stat_ndlist = []
	for i in range(len(search_words)):
		word_list = str(search_words[i]).strip().split(' ')
		class_stat = [0] * search_words = list(df.sex)
class_stat_ndlist = []
for i in range(len(search_words)):
     word_list = str(search_words[i]).strip().split(' ')
     class_stat = [0] * n_class
     if len(word_list) > 0:
          for j in range(len(word_list)):
               word_wgt = str(word_list[j]).strip().split(':')
          if len(word_wgt) > 1:
               word = word_wgt[0]
               cnt = math.ceil(float(word_wgt[1]))
               idx = word_dict_cls.get(word, -1)
          if idx != -1: 
               class_stat[idx] += cnt
     class_stat_ndlist.append(class_stat)
		if len(word_list) > 0:
			for j in range(len(word_list)):
				word_wgt = str(word_list[j]).strip().split(':')
				if len(word_wgt) > 1:
					word = word_wgt[0]
					cnt = math.ceil(float(word_wgt[1]))
					idx = word_dict_cls.get(word, -1)
					if idx != -1: 
						class_stat[idx] += cnt
		class_stat_ndlist.append(class_stat)

	for i in range(n_class):
		ist_class_stat = []
		for j in range(len(search_words)):
			ist_class_stat.append(class_stat_ndlist[j][i])
		data_df['search_word_class_' + str(i)] = ist_class_stat

	# 每类搜索词的占比
	search_var_names = ['search_word_class_' + str(i) for i in range(n_class)]
	data_df['cnt_all_search'] = data_df[search_var_names].sum(axis=1)
	for i in range(len(search_var_names)):
		data_df[search_var_names[i] + '_rate'] = data_df[search_var_names[i]]/data_df.cnt_all_search

	# 风险搜索词统计 ----------------------------------------------
	
	word_dict_risk = {}
	word_dict_risk['search_word_tot_wgt'] = [0] * data_df.shape[0]
	for i in range(len(risk_word_var)):
		word_dict_risk[risk_word[i]] = [0] * data_df.shape[0]

	# 统计每个风险词的wgt
	for i in range(len(search_words)):
		words = str(search_words[i]).strip().split(' ')
		if len(words) > 0:
			tot_wgt = 0
			for j in range(len(words)):
				word_wgt = str(words[j]).strip().split(':')
				if len(word_wgt) > 1:
					tot_wgt += float(word_wgt[1])
					if word_dict_risk.get(word_wgt[0], 'not_found') != 'not_found':
						word_dict_risk[word_wgt[0]][i] += float(word_wgt[1])
			word_dict_risk['search_word_tot_wgt'][i] = tot_wgt
	for i in range(len(risk_word)):
		data_df[risk_word_var[i]] = word_dict_risk[risk_word[i]]

	# 统计风险搜索词总wgt及占比
	data_df['search_word_tot_wgt'] = word_dict_risk['search_word_tot_wgt']
	data_df['tot_risk_search_wgt'] = data_df[risk_word_var].sum(axis=1)
	for i in range(len(risk_word_var)):
		data_df[risk_word_var[i] + '_rate'] = data_df[risk_word_var[i]]/data_df.search_word_tot_wgt
	data_df['rate_tot_risk_search_wgt'] = data_df.tot_risk_search_wgt / data_df.search_word_tot_wgt

	# app分类统计 -------------------------------------------------

	# 统计每类app的安装个数
	apps_list = list(data_df.erised_applist)
	class_stat_ndlist = []
	n_tot_apps = []
	for i in range(len(apps_list)):
		apps = str(apps_list[i]).strip().split(' ')
		n_tot_apps.append(len(apps))
		class_stat = [0] * n_app_class
		for j in range(len(apps)):
			app_cls = app_dict_cls.get(apps[j], 'not_found')
			if app_cls != 'not_found':
				idx = index(app_classes, app_cls)
				if idx != -1: class_stat[idx] += 1
		class_stat_ndlist.append(class_stat)
		
	for i in range(n_app_class):
		ist_class_stat = []
		for j in range(len(apps_list)):
			ist_class_stat.append(class_stat_ndlist[j][i])
		data_df['app_class_' + str(i)] = ist_class_stat
	data_df['n_tot_apps'] = n_tot_apps

	# 每类app安装占比	
	app_cls = ['app_class_' + str(i) for i in range(n_app_class)]	   
	for i in range(len(app_cls)):
		data_df[app_cls[i] + '_rate'] = data_df[app_cls[i]]/data_df.n_tot_apps

	# 风险APP统计 -------------------------------------------------
	
	app_dict_risk = {}
	for i in range(len(risk_app_var)):
		app_dict_risk[risk_app[i]] = [0] * data_df.shape[0]

	# 统计每个风险app是否安装
	for i in range(len(apps_list)):
		apps = str(apps_list[i]).strip().split(' ')
		if len(apps) > 0:
			for j in range(len(apps)):
				if app_dict_risk.get(apps[j], 'not_found') != 'not_found':
					app_dict_risk[apps[j]][i] += 1

	for i in range(len(risk_app)):
		data_df[risk_app_var[i]] = app_dict_risk[risk_app[i]]

	# 统计总的风险app安装个数及占比
	data_df['cnt_risk_app'] = data_df[risk_app_var].sum(axis=1)
	data_df['rate_risk_app'] = data_df.cnt_risk_app / data_df.n_tot_apps

	# LBS ---------------------------------------------------------
	
	lbs_dict = {}
	lbs_dict['lbs_tot_wgt'] = [0] * data_df.shape[0]
	for i in range(len(good_lbs_var)):
		lbs_dict[good_lbs[i]] = [0] * data_df.shape[0]

	lbs_list = list(data_df.erised_user_poi)
	for i in range(len(lbs_list)):
		lbses = str(lbs_list[i]).strip().split(' ')
		if len(lbses) > 0:
			tot_wgt = 0
			for j in range(len(lbses)):
				lbs_wgt = str(lbses[j]).strip().split(':')
				if len(lbs_wgt) > 1:
					tot_wgt += float(lbs_wgt[1])
					if lbs_dict.get(lbs_wgt[0], 'not_found') != 'not_found':
						lbs_dict[lbs_wgt[0]][i] += float(lbs_wgt[1])
			lbs_dict['lbs_tot_wgt'][i] = tot_wgt

	for i in range(len(good_lbs)):
		data_df[good_lbs_var[i]] = lbs_dict[good_lbs[i]]

	data_df['lbs_tot_wgt'] = lbs_dict['lbs_tot_wgt']
	data_df['tot_good_lbs_wgt'] = data_df[good_lbs_var].sum(axis=1)
	for i in range(len(good_lbs_var)):
		data_df[good_lbs_var[i] + '_rate'] = data_df[good_lbs_var[i]]/data_df.lbs_tot_wgt

	data_df['rate_tot_good_lbs_wgt'] = data_df.tot_good_lbs_wgt / data_df.lbs_tot_wgt

	# top 6 兴趣 --------------------------------------------------

	intes_list = [list(data_df.tc1), 
				  list(data_df.tc2), 
				  list(data_df.tc3), 
				  list(data_df.tc4), 
				  list(data_df.tc5), 
				  list(data_df.tc6)]
	cnt_bad_intes = []
	cnt_good_intes = []
	intes_score = []
	top = 6
	for col in range(data_df.shape[0]):
		cnt_bad = 0
		cnt_good = 0
		for row in range(top):
			if intes_list[row][col] in bad_intes: cnt_bad += 1
			if intes_list[row][col] in good_intes: cnt_good += 1
		cnt_bad_intes.append(cnt_bad)
		cnt_good_intes.append(cnt_good)
		intes_score.append(cnt_bad - cnt_good)

	data_df['cnt_bad_intes'] = cnt_bad_intes
	data_df['cnt_good_intes'] = cnt_good_intes
	data_df['stat_bad_minis_good'] = intes_score
	
	# 类别型变量 dummy --------------------------------------------
	
	rm_vars = ['t3',
			   't3_0',
			   'iedu',
			   'iedu_0',
			   'educational_level_0',
			   'erised_applist',
			   'erised_taglist',
			   'erised_user_poi']
	
	data_df = data_df.drop(rm_vars, axis=1)

	cols = data_df.columns.values
	cols_cate = ['tc5_0','reg_prove','job','card_type','card_type_0']
	cols_other = list(set(cols) - set(cols_cate))
	data_df_cate = data_df[cols_cate]
	data_df_o = data_df[cols_other]

	data_df_cate = pd.get_dummies(data = data_df_cate,
							 	  prefix = cols_cate)
	data_df_cate['passid'] = data_df_o.passid
	data_df = pd.merge(data_df_o, data_df_cate, on='passid', how='left')
							 
	try: data_df['tc5_0_旅游出行']	
	except: data_df['tc5_0_旅游出行'] = [0]*data_df.shape[0]
	try: data_df['reg_prove_四川']
	except: data_df['reg_prove_四川'] = [0]*data_df.shape[0]
	try: data_df['job_文职人员']
	except: data_df['job_文职人员']	= [0]*data_df.shape[0]
	try: data_df['card_type_信用卡']
	except: data_df['card_type_信用卡'] = [0]*data_df.shape[0]
	try: data_df['card_type_0_信用卡']
	except: data_df['card_type_0_信用卡'] = [0]*data_df.shape[0]
	
	data_df = data_df[tbl_varname_arr]
	
	return data_df


def output(data_df):
	data_df.replace(to_replace=np.nan, value='', inplace=True)
	data_narr = np.array(data_df, str)
	for i in range(len(data_narr)):
		print ('\t'.join(data_narr[i]))

def line_into_ndarr(line, pass_mon_set, ndarr_cur, ndarr_last):
	if str(cur_mon) in line[1]:
		pass_mon = line[0] + '_' + str(cur_mon)
		if pass_mon not in pass_mon_set:
			ndarr_cur.append(line[0:1] + line[2:])
			pass_mon_set.add(pass_mon)
	elif str(last_mon) in line[1]:
		pass_mon = line[0] + '_' + str(last_mon)
		if pass_mon not in pass_mon_set:
			ndarr_last.append(line[0:1] + line[2:])
			pass_mon_set.add(pass_mon)
	
def main(argv):
	pass_last = None
	pass_mon_set = set()
	ndarr_cur = []
	ndarr_last = []
	
	line = sys.stdin.readline()
	try:
		while line:
			if pass_last == None or pass_last == line.split("\t", 1)[0]:
				line = line.split("\t")
				pass_last = line[0]
				line_into_ndarr(line, pass_mon_set, ndarr_cur, ndarr_last)
			else:
				if len(ndarr_cur) >= n_cache:
					df_cur = pd.DataFrame(data=ndarr_cur, columns=bd_var_names_arr)
					df_last = pd.DataFrame(data=ndarr_last, columns=bd_var_names_arr_lm)
					data_df = pd.merge(df_cur, df_last, on = 'passid', how = 'outer')
					df_cur = None
					df_last = None
					if not data_df.empty:
						data_df = build_var(data_df)
						output(data_df)
					data_df = None
					ndarr_cur = []
					ndarr_last = []
				line = line.split("\t")
				pass_last = line[0]
				line_into_ndarr(line, pass_mon_set, ndarr_cur, ndarr_last)
			line = sys.stdin.readline()
	except "end of file":
		return None

	if pass_last != None:
		df_cur = pd.DataFrame(data=ndarr_cur, columns=bd_var_names_arr)
		df_last = pd.DataFrame(data=ndarr_last, columns=bd_var_names_arr_lm)
		data_df = pd.merge(df_cur, df_last, on = 'passid', how = 'outer')
		df_cur = None
		df_last = None
		if not data_df.empty:
			data_df = build_var(data_df)
			output(data_df)
		data_df = None
		ndarr_cur = []
		ndarr_last = []


if __name__ == "__main__":
	main(sys.argv)
