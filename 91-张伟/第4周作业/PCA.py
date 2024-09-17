import numpy as np


class CPCA(object):

    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  # 样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0] # 样本集的样例总数
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1) # 样本矩阵的协方差矩阵C
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    X = np.array([[12, 12, 42],
                  [13, 32, 13],
                  [11, 11, 32],
                  [23, 32, 56],
                  [43, 12, 43]])
    K = np.shape(X)[1] - 1
    print('样本集(5行3列，5个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)
