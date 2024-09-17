# -*- coding: utf-8 -*-

# 手写PCA算法，求样本矩阵X的K阶降维矩阵Z

import numpy as np
import random

# X: 样本矩阵，m行n列，m=样本数量，n=特征维度
# k: X降维后的阶数
class CPCA(object):
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.centerX = [] # X的中心化
        self.C = []      # X的协方差矩阵C
        self.U = []      # X的降维转换矩阵，由被选中的特征值对应的特征向量组成
        self.Z = []      # 最终X的降维矩阵Z

        # PCA算法开始，分为3个步骤：
        # step1. 原始数组的中心化
        self.centerX = self._centralized()

        # step2. 求X的协方差矩阵
        self.C = self._cov()

        # step3. 求降维矩阵Z
        self.U = self._U() # step3.1 先求转换矩阵U
        self.Z = self._Z() # step3.2 Z = XU 求得最终结果



    # 矩阵的中心化，即减去平均值
    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centerX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) # 计算平均值，其中self.X.T是矩阵转置
        print('X的特征均值:\n', mean) # [28.8 31.5 26.7]
        centerX = self.X - mean # 中心化
        print('X的中心化centerX:\n', centerX)
        return centerX
    
    # 求X的协方差矩阵
    def _cov(self):
        ns, nw = np.shape(self.centerX) # 样本数量，即m
        print('样本数量: ', ns) # 10
        C = np.dot(self.centerX.T, self.centerX)/(ns - 1) # 因为Step1中做过中心化，所以协方差矩阵简化为这样求
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C
    
    # 求转换矩阵U
    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        eva, evt = np.linalg.eig(self.C) # 求X的协方差矩阵C的特征值和特征向量
        print('样本集的协方差矩阵C的特征值:\n', eva)
        print('样本集的协方差矩阵C的特征向量:\n', evt)

        ind = abs(np.argsort(-eva)) # 将特征值降序排列，注意这里要用argsort，不用sort，因为是对索引进行排序
        print('降序排列后的特征值: ', ind) # [2 1 0]

        # 构建K阶降维的降维转换矩阵U，下面这行对我来说有点复杂T_T
        # UT = [evt[:,ind[i]] for i in range(self.k)]

        # 下面这几都是构建K阶降维的降维转换矩阵U，简洁的写法后面再看
        ns, nw = np.shape(self.centerX)
        print('X的特征维度总: ', nw)
        UT = np.zeros((self.k, nw), dtype='double') # 转换矩阵的shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        for i in range(self.k):
           _index = ind[i]
           UT[i] = evt[_index]
        U = np.transpose(UT)

        print('%d阶降维转换矩阵U:\n'%self.k, U)
        return U
    
    # 按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape: ', np.shape(self.X))
        print('U shape: ', np.shape(self.U))
        print('Z shape: ', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
    

# 生成m行，n列的(0-100)随机数组
def genTestArr(m, n, _range):
    _result = np.zeros((m, n), dtype = int)

    for i in range(m):
        for j in range(n):
            _item = [random.randint(0, _range), random.randint(0, _range), random.randint(0, _range)]
        _result[i] = _item

    
    return _result

if __name__ == '__main__':
    X = genTestArr(10, 3, 50) # 生成一个10个样本(行数)、3个特征维度(列数)、取值范围0-50的随机数组
    k = np.shape(X)[1] - 1 # 降1个维度
    pca = CPCA(X, k)
