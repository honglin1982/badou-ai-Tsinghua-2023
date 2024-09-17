# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

#自己实现PCA
class PCA():
    '''
    用PCA求样本矩阵X的降维矩阵Z
    NOTE:请保证输入的样本矩阵X shape=(m,n)，m行样例，n个特征
    '''
    def __init__(self,X,K):
        '''
        :param X 训练样本矩阵
        :param k 降维后的阶数
        '''
        self.X = X       #样本矩阵X
        self.K = K       #k阶降维后的阶数
        self.centrX = [] #样本矩阵的中心化
        self.C = []      #样本集的协方差矩阵
        self.U = []      #样本矩阵X的降维转换矩阵
        self.Z = []      #样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C      = self._cov()
        self.U      = self._U()
        self.Z      = self._Z() #Z=XU求得

    def _centralized(self):
            '''矩阵X的中心化'''
            print('样本矩阵X：\n',self.X)
            centrX = []
            mean = np.array([np.mean(attr) for attr in self.X.T]) #样本集的特征均值
            #mean = np.mean(self.X,0)
            centrX = self.X - mean #样本集中心化
            print('样本矩阵X的中心化centrX:\n',centrX)
            return centrX
    def _cov(self):
            '''求样本矩阵X的协方差矩阵C'''
            #样本集的样例总数
            ns = np.shape(self.centrX)[0]
            print('样本集的样例总数ns',ns)
            # 样本矩阵的协方差矩阵C
            C = np.dot(self.centrX.T, self.centrX) / ns
            print('样本矩阵X的协方差矩阵C:\n',C)
            return C
    def _U(self):
            '''求X的降维转换矩阵U，shape=(n,k),n是X的特征维度总数，k是降维矩阵的特征维度'''
            #先求x的协方差矩阵C的特征值和特征向量
            a,b = np.linalg.eig(self.C) #a为特征值，b为特征向量
            print('样本集的协方差矩阵C的特征值:\n', a)
            print('样本集的协方差矩阵C的特征向量:\n', b)
            #给出特征值降序的topK的索引序列
            ind = np.argsort(-1*a)
            print('索引序列ind',ind)
            # 构建K阶降维的降维转换矩阵U
            UT = [b[:, ind[i]] for i in range(self.K)]
            #UT = b[:ind[self.K]]
            print('%d阶降维转换矩阵UT:\n' % self.K, UT)
            U = np.transpose(UT)
            print('%d阶降维转换矩阵U:\n' % self.K, U)
            return U
    def _Z(self):
            '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
            Z = np.dot(self.X, self.U)
            print('X shape:', np.shape(self.X))
            print('U shape:', np.shape(self.U))
            print('Z shape:', np.shape(Z))
            print('样本矩阵X的降维矩阵Z:\n', Z)
            return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    # X = np.array([[10, 15, 29],
    #               [15, 46, 13],
    #               [23, 21, 30],
    #               [11, 9,  35],
    #               [42, 45, 11],
    #               [9,  48, 5],
    #               [11, 21, 14],
    #               [8,  5,  15],
    #               [11, 12, 21],
    #               [21, 20, 25]])
    X,y= load_iris(return_X_y=True)
    K = np.shape(X)[1] - 2
    print('K = ', K)
    pca = PCA(X,K)

    #可视化
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(pca.Z)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(pca.Z[i][0])
            red_y.append(pca.Z[i][1])
        elif y[i] == 1:
            blue_x.append(pca.Z[i][0])
            blue_y.append(pca.Z[i][1])
        else:
            green_x.append(pca.Z[i][0])
            green_y.append(pca.Z[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

#sklearn方式
    # pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
    # reduced_x = pca.fit_transform(X)  # 对原始数据进行降维，保存在reduced_x中
    # print('样本矩阵X的降维矩阵reduced_x:\n', reduced_x)
    # red_x, red_y = [], []
    # blue_x, blue_y = [], []
    # green_x, green_y = [], []
    # for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
    #     if y[i] == 0:
    #         red_x.append(reduced_x[i][0])
    #         red_y.append(reduced_x[i][1])
    #     elif y[i] == 1:
    #         blue_x.append(reduced_x[i][0])
    #         blue_y.append(reduced_x[i][1])
    #     else:
    #         green_x.append(reduced_x[i][0])
    #         green_y.append(reduced_x[i][1])
    # plt.scatter(red_x, red_y, c='r', marker='x')
    # plt.scatter(blue_x, blue_y, c='b', marker='D')
    # plt.scatter(green_x, green_y, c='g', marker='.')
    # plt.show()