import cv2
import numpy as np


"""
PCA的主要步骤如下：
1、对原始数据进行去均值处理，即将每个特征的均值减去。
2、计算数据的协方差矩阵。
3、对协方差矩阵进行特征值分解，得到特征值和对应的特征向量。
4、根据特征值的大小对特征向量进行排序，选取前k个特征向量作为主成分。
5、将原始数据投影到选取的主成分上，得到降维后的数据。
"""

"""
使用PCA求样本矩阵 S 的 K阶降维矩阵 Z
"""

"""
用PCA求样本矩阵 X 的 K阶降维矩阵Z
Note:请保证输入的样本矩阵满足 shape=(m, n)，m行样例，n个特征
"""
class CPCA(object):
    def __init__(self, X, K):
        self.X = X   # 样本矩阵X
        self.K = K   # K阶降维矩阵的K值
        self.center_X = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.center_X = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        """样本矩阵X的中心化处理"""
        print('\n样本矩阵X:\n', self.X)
        # np.mean 对样本集的特征进行均值计算，axis=0 表示按列计算均值，得到每个特征的均值
        characteristic_mean = np.mean(self.X, axis=0)
        print('样本集的特征均值为:\n', characteristic_mean)

        # 对样本集做0均值化（中心化）,样本集中每个特征减去该特征的平均值
        center_X = self.X - characteristic_mean
        print('样本矩阵X的中心化center_X:\n', center_X)
        return center_X

    def _cov(self):
        """求样本矩阵X的协方差矩阵C"""
        # 样本集的样例总数，m 为样本
        sample_number = self.center_X.shape[0]
        # 求样本矩阵的协方差矩阵C: 每个样本向量与其转置的乘积，并对所有样本向量求和, 然后除以 (m-1) 得到协方差矩阵。
        C = np.dot(self.center_X.T, self.center_X) / (sample_number - 1)
        print('\n样本矩阵X的协方差矩阵C:\n', C)   # C 为（3，3）
        return C

    def _U(self):
        """求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度"""
        # 计算协方差矩阵C的特征值数组和对应的特征向量矩阵, C为方阵
        eigenvalues, eigenvectors = np.linalg.eig(self.C)
        print('\n样本集的协方差矩阵C的特征值:\n', eigenvalues)
        print('样本集的协方差矩阵C的特征向量:\n', eigenvectors)

        # 根据特征值的降序排列获取索引序列, 加负号是为了降序排列
        indices = np.argsort(-eigenvalues)
        print(f"特征值的降序排列后的索引序列为: {indices}")

        # 构建K阶降维的降维转换矩阵U, eigenvectors[:, indices[:self.K]] 表示从特征向量矩阵中选取所有行，以及降序排列后的前 self.K 个列（特征向量）
        U = eigenvectors[:, indices[:self.K]]
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        """按照Z=XU求降维矩阵Z, shape=(m,k), m是样本总数，k是降维矩阵中特征维度总数"""
        # 两个矩阵相乘的条件是，第一个矩阵的列数必须等于第二个矩阵的行数。
        Z = np.dot(self.X, self.U)
        print('\nX shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    # 10样本3特征的样本集, 行为样例，列为特征维度
    X = np.random.randint(low=0, high=51, size=(10, 3))
    print(f"样本集(10行3列，10个样例，每个样例3个特征):\n{X}")

    K = np.shape(X)[1] - 1

    pca = CPCA(X, K)



