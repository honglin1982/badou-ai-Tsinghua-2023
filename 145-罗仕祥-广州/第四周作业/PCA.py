#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris
import numpy as np


# 自己实现的PCA
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        '''矩阵X的中心化'''
        self.centerX = X - X.mean(axis=0)
        '''求协方差矩阵'''
        self.covariance = np.dot(self.centerX.T, self.centerX) / (X.shape[0] - 1)
        '''求协方差矩阵的特征值、特征向量'''
        w, v = np.linalg.eig(self.covariance)
        '''特征值降序排列索引'''
        ind = np.argsort(-w)
        '''构建特征向量矩阵'''
        self.components_ = v[:, ind[:self.n_components]]
        '''求样本集的降维矩阵'''
        df = np.dot(self.centerX, self.components_)
        return np.dot(self.centerX, self.components_)

# sklearn实现的PCA函数
class PCA_sklearn():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # X: 原始数据
        pca = dp.PCA(n_components=self.n_components)
        return pca.fit_transform(X)


# 读取数据
x, y = load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签

# 降维
pca = PCA(n_components=2)
# pca = PCA_sklearn(n_components=2)
reduced_x = pca.fit_transform(x) #对原始数据进行降维，保存在reduced_x中

# 可视化
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
# print("降维后的样本集：\n", reduced_x)
# print("PCA_sklearn特征向量：\n",pca.components_)
