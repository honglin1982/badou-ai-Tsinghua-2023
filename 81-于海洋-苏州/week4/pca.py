# -*- coding: utf-8 -*-
"""
@author: 81-于海洋
第四周 PCA
"""

import numpy as np


class Pca(object):
    def __init__(self, src_array, dimension):
        """
        :param src_array 样本矩阵
        :param dimension 目标维度
        """
        self.src = src_array
        self.dimension = dimension
        # 零均值化
        self.center_arr = []
        # 协方差
        self.cov_arr = []
        self.linalg_arr = []  # 样本矩阵X的降维转换矩阵
        self.result_arr = []  # 样本矩阵X的降维矩阵Z
        print("原始矩阵:\n", self.src)
        self.center_arr = self.centralization()
        print("零均值化:\n", self.center_arr)
        self.cov_arr = self.cov()
        print("协方差后:\n", self.cov_arr)
        self.linalg_arr = self.linalg()
        print('%d阶降维转换矩阵:\n' % self.dimension, self.linalg_arr)
        self.result_arr = self.final()  # Z=XU求得
        print('最终结果:\n', self.result_arr)

    def centralization(self):
        """
        Step1. 矩阵零均值化 （中心化）
        中心化指的是 变量减去他的均值使均值为0 => 平移过程，平移后所有数据的中心是 0
        原均值：（x1 + x2 + x3 + .. + xn）/ n  = E
        零均值： （x1-E + x2-E + xn-E） / n
            =>  (x1 + x2 + xn - nE) / n
            =>  (x1 + x2 +..xn) / n - nE/n
            =>  E - E
            =>  0

        例如：
        [3, 1]  =>  2  => [3-2, 1-2]  => [1, -1]
        """

        """
        self.src.T 取二维数组的值 例如：
        [[1, 1], [2, 2]] 那么每次取值 分别为 [1, 1]  [2, 2]  
        
        np.mean 计算平均值
        """
        mean = np.array([np.mean(row) for row in self.src.T])
        print('均值数据:', mean)
        return self.src - mean

    def cov(self):
        """
        Step2. 求协方差矩阵 cov
        协方差：度量两个随机变量关系的统计量。
        ***  同一元素的方差 表示该元素的方差 ***
        ***  不同元素之间的协方差就是他们的相关新 ***

        cov(x,y) > 0  =>  X与Y正相关
        cov(x,y) < 0  =>  X与Y负相关
        cov(x,y) = 0  =>  X与Y不相关

        方差公式：Var(X) = Σ((X - μ)^2) / N
        X：随机变量，包含 N 个观测值。
        μ：随机变量 X 的平均值。
        (X - μ)^2：将每个观测值与平均值的差的平方。
        Σ：求和符号，表示对所有观测值进行求和。
        / N：将求和的结果除以观测值的总数 N。


        协方差公式：Cov(X, Y) = Σ((X - μX) * (Y - μY)) / N
        X：随机变量 X，包含 N 个观测值。
        Y：随机变量 Y，包含 N 个观测值。
        μX：随机变量 X 的平均值。
        μY：随机变量 Y 的平均值。
        (X - μX)：X 中每个观测值与 X 的平均值之差。
        (Y - μY)：Y 中每个观测值与 Y 的平均值之差。
        (X - μX) * (Y - μY)：将 X 和 Y 中对应观测值的差相乘。
        Σ：求和符号，表示对所有观测值进行求和。
        / N：将求和的结果除以观测值的总数 N。
        """
        shape = np.shape(self.center_arr)[0]
        # np.dot 样本乘法  公式：：Cov(X, Y) = Σ((X - μX) * (Y - μY)) / N
        # self.center_array.T = X 均值差
        # self.center_array   = Y 均值差
        return np.dot(self.center_arr.T, self.center_arr) / (shape - 1)

    def linalg(self):
        """
        Step3. 求特征向量和特征值
        求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        """
        # 先求X的协方差矩阵C的特征值和特征向量
        # 特征值赋值给a，对应特征向量赋值给b。函数doc：
        # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        n, k = np.linalg.eig(self.cov_arr)
        print('特征维度总数:\n', n)  # 特征值
        print('特征维度:\n', k)  # 特征向量
        # 给出特征值降序的topK的索引序列
        # np.argsort 返回排序后的索引（下标值）
        # arr = np.array([3, 1, 4, 2, 5])
        # sorted_indices = np.argsort(arr)
        # print(sorted_indices)  # 输出：[1 3 0 2 4]
        ind = np.argsort(-1 * n)
        # 构建K阶降维的降维转换矩阵U ？
        UT = [k[:, ind[i]] for i in range(self.dimension)]
        # 数据转置
        return np.transpose(UT)

    def final(self):
        """按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数"""
        result = np.dot(self.src, self.linalg_arr)
        print('src shape:', np.shape(self.src))
        print('降维 shape:', np.shape(result))
        return result


if __name__ == '__main__':
    """ 
    2023/05/25 理解
    主成分分析：PCA -> 降维操作 
    
    目标：
    1. 降维后同一纬度方差最大
    2. 不同维度之间的相关性为0
    
    主要步骤：
    1. 对原始数据零均值化（中心化）
    2. 求协方差矩阵
    3. 对协方差矩阵求特征向量和特征值。（特征向量的矩阵->特征矩阵） 
    
     例如： 一个 3 x 3 的矩阵 降维  2 x 3 矩阵  X Y Z M N 为特征信息。 
         X   Y   Z               M   N
    A |  1   1   3         A  |  6   4
    B |  2   3   1    =>   B  |  3   7
    C |  3   5   6         C  |  5   5
    
    """
    "原始矩阵"
    src = np.array([
        [10, 11, 12],
        [11, 11, 11],
        [12, 11, 10]
    ])

    "目标维度-> 降一维度"
    target_dimension = np.shape(src)[1] - 1
    pca = Pca(src, target_dimension)
