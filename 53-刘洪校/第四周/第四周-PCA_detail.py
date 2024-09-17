# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:39:56 2023

@author: lhx
"""
#PCA步骤
#1. 对原始数据零均值化（中心化），
#2. 求协方差矩阵，
#3. 对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间。
import numpy as np
#'10样本3特征的样本集, 行为样例，列为特征维度'
arr=np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9,  35],
              [42, 45, 11],
              [9,  48, 5],
              [11, 21, 14],
              [8,  5,  15],
              [11, 12, 21],
              [21, 20, 25]])
#K阶降维矩阵的K值
K = np.shape(arr)[1] - 1

#1. 对原始数据零均值化（中心化）求每一个特征的平均值，然后对于所有的样本，每一个特征都减去自身的均值。
#1.1 求样本集的特征均值
mean=np.mean(arr,0)## axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
#mean=np.array([np.mean(attr) for attr in arr.T])
print("样本集的特征均值:\n",mean)
#1.2 样本集的中心化
centerArr=arr-mean
print("样本集的中心化:\n",centerArr)

#2. 求协方差矩阵
#ps:方差->相减后平方
#2.1 样本集的样例总数
ns = np.shape(centerArr)[0]
print(ns)
print(len(centerArr))

#2.2 样本矩阵的协方差矩阵C
#covArr=np.dot(centerArr.T, centerArr)/(ns - 1)
covArr=np.cov(centerArr.T)
print('样本的协方差矩阵C:\n', covArr)

#3.对协方差矩阵求特征向量和特征值
# np.linalg.eig 它将返回两个值，第一个是特征值，第二个是特征向量。
a,b=np.linalg.eig(covArr)
print('样本集的协方差矩阵的特征值:\n', a)
print('样本集的协方差矩阵的特征向量:\n', b)

#给出特征值降序的topK的索引序列
#索引排序，-1正序
ind = np.argsort(-1*a)
print(ind)
# for i in range(K):
#     print(b[:,ind[i]])
UT = [b[:,ind[i]] for i in range(K)]#,取特征向量的前K列
print(UT)
U = np.transpose(UT)
print('%d阶降维转换矩阵U:\n'%K, U)

'''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
Z = np.dot(arr, U)
print('X shape:', np.shape(arr))
print('U shape:', np.shape(U))
print('Z shape:', np.shape(Z))
print('样本矩阵X的降维矩阵Z:\n', Z)
