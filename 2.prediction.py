
###############################################################################################################################################################
######################################################################### 模型分修正：data_set_score #####################################################################
######## 1、

xpath='/home/fsg/jiwenchao/ac_class/data'
model_name='/01_ac_class_fraud.model'

# read data 

na_values=['','NULL','NA','null','na','Na','-9999','-1','Infinity','NaN']

data_set = pd.read_table('/home/fsg/jiwenchao/ac_class/data/data_set_0705.txt',sep = '\t',na_values = na_values)

# selected model vars 
col_x_old = list(data_set.columns)

#临时删除关联字段
remove_vars=['passid','sessionid','data_set_tag','sessionid.1','risk_tag_union','paytype_credit_pay','uid_base_payamt']

col_x = [x for x in col_x_old if x not in remove_vars]


data_set_all = data_set[col_x].as_matrix()
data_prob_value = xgb.DMatrix(data_set_all)


# load model 
bst = xgb.Booster()
bst.load_model(xpath+model_name)



#保留三列对应值
pred_df = data_set[['sessionid','passid','risk_tag_union']]
pred_df['score'] = bst.predict(data_prob_value)  #添加一列score，注意ddafen数据的类型，DMatrix


#保留数据，用于模型分修正
pred_df.to_csv('/home/fsg/jiwenchao/ac_class/data/data_set_score.txt',sep = '\t', header = True, index = False,encoding = 'utf-8')
