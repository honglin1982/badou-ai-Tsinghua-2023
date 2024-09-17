# -*- coding: utf-8 -*-

import numpy as np


# 定义一个PCA的类
class CPCA:

    # 定义一个构造函数，参数（X样本矩阵 K是想要的特征值）、各个变量（中心化矩阵、C协方差矩阵、U降维转换矩阵、Z降维矩阵）
    def __init__(self, X, K):
        self.X = X
        self.K = K
        # 变量占位
        self.centralX = []
        self.C = []
        self.U = []
        self.Z = []
        # 变量运用函数的占位
        self.centralX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    # 矩阵中心化函数，np.array+np.mean()求均值构成的矩阵，np.shape()取行列数
    def _centralized(self):
        print("样本矩阵为：\n", X)
        centralX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print("样本集的特征均值为：\n", mean)
        centralX = self.X - mean
        print("样本矩阵去中心化矩阵后为：\n", centralX)
        return centralX

    # 协方差矩阵函数D=Xt*X/m-1
    def _cov(self):
        m = np.shape(self.centralX)[0]
        C = np.dot(self.centralX.T, self.centralX)/(m - 1)
        print("样本矩阵的协方差矩阵C为：\n", C)
        return C

    # 降维转换矩阵函数，用np.linalg.eig(X)求特征值a、特征向量b。特征值从大到小排序(取索引值)的K个特征向量的组合。
    def _U(self):
        a, b = np.linalg.eig(self.C)
        print("样本集的协方差矩阵的特征值为：\n", a)
        print("样本集的协方差矩阵的特征向量为：\n", b)
        ind = np.argsort(-1*a)
        UT = [b[:, ind[j]] for j in range(self.K)]
        U = np.transpose(UT)
        print("%d阶降维转换矩阵U为:\n"%self.K, U)
        return U

    # 降维矩阵Z=Xnew*U
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print("X shape:", np.shape(self.X))
        print("U shape:", np.shape(self.U))
        print("Z shape:", np.shape(Z))
        print("样本矩阵的降维矩阵为：\n", Z)
        return Z


if __name__ == "__main__":
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print("样本集（10行3列，10个样本，每个样本有三个特征）：\n", X)
    X_pca = CPCA(X, K)

# 转置矩阵：1.X.T  2.np.transppose(X)
# 求均值：np.mean(arr,axis)
# 行列数：np.shape(X)
# 行数：np.shape(X)[0]  列数：np.shape(X)[1]
# np.argsort(x) 默认将x从小到大排序，并输出索引值
