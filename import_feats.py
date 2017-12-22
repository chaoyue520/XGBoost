#!/usr/bin/python
#-*- coding:utF-8 -*-


# 跑完xgb模型以后，输出模型bst和fmap列表，变量重要性计算指标为 gain*weight
def fea_importance(xgb_obj, mapfile, resfile):
    importance_weight = xgb_obj.get_score(fmap=mapfile, importance_type='weight')
    importance_gain = xgb_obj.get_score(fmap=mapfile, importance_type='gain')
    fout = open(resfile, 'w')
    for fea in importance_weight:
        weight = importance_weight[fea]
        gain = importance_gain[fea]
    fout.write('\t'.join(map(str, [fea.encode('utf-8'), weight, gain, '\n'])))
    return 0


fea_importance(bst, 'test_1222.txt', 'fea_importance_file')