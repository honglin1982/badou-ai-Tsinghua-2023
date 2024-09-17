#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np


class PCA_Details(object):
    def __init__(self,X,K):

        '输入矩阵shape=(m,n)，m个样本，n个特征'

        self.X = X          #   样本矩阵
        self.K = K          #   K阶降维矩阵
        self.centrX = []  #   矩阵X的中心化
        self.C = []         #   矩阵的协方差矩阵C
        self.U = []         #   矩阵X的变换矩阵U
        self.Z = []         #   矩阵X的降维矩阵Z

        self.centrX = self._centralize()
        self.C = self._cov()
        self.U = self._U()
        self.z = self._Z()
    def _centralize(self):
        '''矩阵X的中心化'''
        print("样本矩阵X为：",self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        print("样本矩阵X的中心化矩阵centrx：",centrX)
        return centrX

    def _cov(self):
        ''' 求样本矩阵X的协方差矩阵U'''
        num = np.shape(self.centrX)[0]      #   求的矩阵的样本数量
        C = np.dot(self.centrX.T,self.centrX)/(num-1)
        print('样本矩阵X的协方差矩阵C：\n',C)
        return C

    def _U(self):
        a,b = np.linalg.eig(self.C)      #   求的样本集的协方差矩阵的的特征值和特征向量
        print("样本集的协方差矩阵的特征值为：\n",a)
        print("样本集的协方差矩阵的特征向量为：\n",b)
        ind = np.argsort(-1*a)      #   对特征值按降序排列
        UT = [b[:,ind[i]] for i in range(self.K)]   #  构建降维转换矩阵
        U = np.transpose(UT)
        print("降维矩阵U为：",U)
        return U

    def _Z(self):
        Z = np.dot(self.X,self.U)       #   按照Z=XU求降维矩阵
        print("X shape:",np.shape(self.X))
        print("U shape:",np.shape(self.U))
        print("Z shape:",np.shape(Z))
        print("样本矩阵X的降维矩阵为：\n",Z)
        return Z


if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print("样本集(10行3列，10个样例，每个样例3个特征):\n", X)
    pca = PCA_Details(X,K)



