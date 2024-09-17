import numpy as np

"""
PCA 其实就是降维
1，中心化: 更好的反映数据的离散特征
2，协方差矩阵: 因为pca作用是降维，则是将不同维度的特征不断减少，而且不同维度特征互不相关，正好协方差矩阵反映的就是属性间的关系，也就是协方差矩阵=0。反映的是不同维度之间的X与Y的不相关性
"""


class PyToPCA:
    '''
    用PCA求样本矩阵X的K阶降维矩阵Z
    Note: 请保证输入的样本矩阵X
    shape = (m, n)，m行样例，n个特征
    '''

    def __init__(self, x: np.ndarray, k: int):
        """
        :param x: 训练样本矩阵X
        :param k: K,X的降维矩阵的阶数，即X要特征降维成k阶
        """
        self.X = x  # 矩阵样本
        self.K = k  # K阶降维矩阵的K值
        self.center_X = self.center_func()  # 中心化矩阵  # 中心化矩阵
        self.C = []  # 协方差矩阵
        self.U = []  # 计算得出的降维转换矩阵
        self.Z = []  # 输出样本矩阵X的降维矩阵

    def center_func(self):
        """ 中心化矩阵 """
        avg = np.array([np.mean(attr) for attr in self.X.T])
        # print(f'---均值：{avg}')
        center_x = self.X - avg  # 样本减均值
        # print(f'---中心化矩阵: \n{center_x}')
        return center_x

    def _covariance_matrix(self):
        """ 协方差矩阵
        公式：Z**t * Z / m
        """
        d = np.dot(self.center_X.T, self.center_X) / (self.center_X.shape[0] - 1)
        # print(f'---协方差变为了对称矩阵: \n{d}')
        return d

    def _reduce_level_c(self):
        """ 协方差矩阵求出特征值和特征向量 """
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        # print('---样本集的协方差矩阵C的特征值:\n', a)
        # print('---样本集的协方差矩阵C的特征向量:\n', b)
        index_sort = np.argsort(-1 * a)  # 给出特征值排序前的topK的索引序列
        # print(f'index_sort: {index_sort}')
        u = np.array([b[:, index_sort[i]] for i in range(self.K)]).T  # 筛选特征，构建K阶降维的降维转换矩阵U, 结果需要转置
        # print('%d阶降维转换矩阵U:\n' % self.K, u)
        return u

    def _fit(self):
        self.C = self._covariance_matrix()  # 协方差矩阵
        self.U = self._reduce_level_c()  # 降维
        self.Z = z = np.dot(self.X, self.U)  # Z = XU求得  样本*特征
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(z))
        print('样本矩阵X的降维矩阵Z:\n', z)
        return self.Z


if __name__ == '__main__':
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
    # print(np.shape(X))
    K = np.shape(X)[1] - 1
    # print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PyToPCA(X, K)
    pca._fit()


