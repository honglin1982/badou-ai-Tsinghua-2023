# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets._base import load_iris
import matplotlib.pyplot as plt


def pcaDetail(X, k):
    # 一. 对原始数据零均值化（中心化）：变量减去它的均值，使均值为0
    # mean(a, axis)求平均值，a必须为数组，axis 不设置，对m*n个数求平均值，返回的是一个实数；
    # axis=0：压缩行对各列求平均值，返回1*n的矩阵；
    # axis=1：压缩列对各行求平均值，返回m*1的矩阵。
    # 1. 求均值
    meanArr = np.mean(X, axis=0)
    print('求得各列的均值', meanArr)
    # 2. 变量减去均值
    centerX = X - meanArr
    print('中心化后的样本: \n', centerX)

    # 二. 求协方差矩阵
    # 中心化矩阵的协方差矩阵公式：D = (Z^T · Z) / (m-1)
    # 当样本数较大时，不必在意其是 m 还是 m-1，为了方便计算，我们分母取 m。
    # np.transpose函数是对矩阵进行转置，也可使用属性 centerX.T
    centerX_T = np.transpose(centerX)
    D = np.dot(centerX_T, centerX) / (centerX.shape[0] - 1)
    print('协方差矩阵：\n', D)

    # 三. 对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间
    # 1. 求得协方差矩阵特征值和特征向量
    # numpy.linalg.eig()函数。
    # 该函数返回一个元组，其中第一个元素是包含特征值的一维数组，
    # 第二个元素是包含特征向量的二维数组。
    a, b = np.linalg.eig(D)
    print("特征值：", a, "\n特征向量：\n", b)
    # 2. 对特征值进行排序
    # np.argsort()函数返回的是数组值从小到大的索引值
    idx = np.argsort(a)[::-1]
    W = []
    for i in range(k):
        W.append(b[:, idx[i]])
    # 将矩阵的行转换为列（转置）
    W_T = np.transpose(W)
    # 降维计算 X · W_T
    Z = np.dot(X, W_T)
    print("降维后的结果：\n", Z)
    return Z


def showData(plt, Z, y):
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(Z)):
        if y[i] == 0:
            red_x.append(Z[i][0])
            red_y.append(Z[i][1])
        if y[i] == 1:
            blue_x.append(Z[i][0])
            blue_y.append(Z[i][1])
        if y[i] == 2:
            green_x.append(Z[i][0])
            green_y.append(Z[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')


if __name__ == '__main__':
    # 10样本3特征的样本集, 行为样例，列为特征维度
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
    # 手动实现PCA降维
    pcaDetail(X, X.shape[1] - 1)

    # PCA api
    # 降到2维
    pca = PCA(n_components=2)
    # 拟合
    pca.fit(X)
    # 获取降维后的数据
    X_new = pca.fit_transform(X)
    print('降维后的结果: \n', X_new)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 加载鸢尾花数据集，x是样本，y是标签
    x, y = load_iris(return_X_y=True)
    # 手动实现PCA对鸢尾花数据集降维
    Z = pcaDetail(x, 2)
    ax = plt.subplot(121)
    ax.set_title("手动实现PCA")
    showData(ax, Z, y)

    # 使用API进行鸢尾花数据集降维
    pca = PCA(n_components=2)
    pca.fit(x)
    Z = pca.fit_transform(x)
    ax = plt.subplot(122)
    ax.set_title("PCA API")
    showData(ax, Z, y)
    plt.show()
