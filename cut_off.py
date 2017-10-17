

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

na_values = ['', 'NA', 'na', 'NULL', 'null', 'NaN', 'nan', '\\N']

def get_lift(data_file, n_groups, output_file, y_name='y', score_name='score'):
	'''
	data_file: 要带表头，\t分割
	'''
	
	# load data

	data = pd.read_table(data_file, na_values = na_values)
						 
	# sort by score
	data_sorted = data.sort_values(by = score_name, axis = 0, ascending = False)  # 按照欺诈得分降序排列

	# group and stat
	n_rcds = data_sorted.shape[0]
	n_rcd_p_group = int(float(n_rcds)/n_groups)
	n_fraud_all = data_sorted[y_name].sum()	# 总欺诈数
	stat = []
	for i in range(n_groups):
		y_this_g = pd.Series(np.array(data_sorted[y_name])[i*n_rcd_p_group : (i+1)*n_rcd_p_group])
		score_this_g = pd.Series(np.array(data_sorted[score_name])[i*n_rcd_p_group : (i+1)*n_rcd_p_group])
		
		id = i+1
		# 分组模型平均分
		avg_score_g = score_this_g.dropna().mean()
		# 分组欺诈数
		n_fraud_g = y_this_g.dropna().sum()
		# 分组欺诈率
		r_fraud_g = float(n_fraud_g) / n_rcd_p_group
		# 每组的最后一个分数
		last_score_g = score_this_g.dropna().min()
		stat.append([id, n_fraud_all, n_rcd_p_group, avg_score_g, n_fraud_g, r_fraud_g, last_score_g])

	# 累计捕获率
	r_capture_acc = []
	acc = 0
	n = len(stat)
	for i in range(n):
		acc += float(stat[i][4])/stat[i][1]
		r_capture_acc.append(acc)

	# output lift result
	header = ['分组', '总欺诈人数', '每组人数', '每组平均分', '每组欺诈人数', '每组欺诈率', '每组最小分', '累计捕获率']
	n_col = len(stat[0])
	with open(output_file, 'wt') as f:
		f.write('\t'.join(header) + '\n')
		for i in range(len(stat)):
			for j in range(n_col):
				f.write(str(stat[i][j]) + '\t')
			f.write(str(r_capture_acc[i]) + '\n')
			

if __name__ == '__main__':
	data_file = 'ac_class_score_new.txt'
	y_name='risk_tag_union'
	score_name='score_new'
	n_groups = 10000
	output_file = 'ac_class_lift_result.txt'
	get_lift(data_file, n_groups, output_file, y_name, score_name)
