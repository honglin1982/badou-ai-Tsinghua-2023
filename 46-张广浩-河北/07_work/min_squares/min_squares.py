# -*- coding: utf-8 -*-
# @Time    : 2023/6/16 9:05
# @Author  : zgh
# @FileName: min_squares.py
# @Software: PyCharm

import pandas as pd
import numpy as np

sales = pd.read_csv('../min_squares/train_data.csv',sep='\s*,\s*',engine='python')

X = sales['X'].values
Y = sales['Y'].values

# 初始化赋值
sum_xy = 0
sum_x = 0
sum_y = 0
sum_xx = 0
n = len(X)

sum_xy = np.dot(X, Y)
sum_x = np.sum(X)
sum_y = np.sum(Y)
sum_xx = np.sum(np.square(X))

# for i in range(n):
#     s1 = s1 + X[i]*Y[i]     #X*Y，求和
#     s2 = s2 + X[i]          #X的和
#     s3 = s3 + Y[i]          #Y的和
#     s4 = s4 + X[i]*X[i]

# 计算斜率和截距
k = (sum_x*sum_y-n*sum_xy)/(sum_x*sum_x-sum_xx*n)
b = (sum_y - k*sum_x)/n
print("k: {} b: {}".format(k, b))
# y=1.4x+3.5
