"""
xgboost package pratice
"""
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

##Data api
#使用xgb.DMatrix()实现数据类型的转换

#1.libsvm数据类型和二进制缓存本地文件
dataset1 = xgb.DMatrix('data.svm.txt')
dataset2 = xgb.DMatrix('data.svm.buffer')

#2.加载numpy数组
data = np.random.rand(5, 10)
label = np.random.randint(2, size=5)
dataset3 = xgb.DMatrix(data, label=label)

#3. DMatrix转换成二进制保存，加载时提高加载速度
dataset1.save_binary('dataset1.buffer')

#4. 在转DMatrix时可以使用缺失值补齐或设置权重等参数设置
weight = np.random.rand(5, 1)
dataset4 = xgb.DMatrix(data, label, missing = -999.0, weight=weight)

