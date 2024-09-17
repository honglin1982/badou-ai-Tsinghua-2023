import numpy as np


class CPCA:
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centerX = []
        self.c = []
        self.u = []
        self.z = []
        self.centerX = self._centerx()
        self.c = self._cov()
        self.u = self._u()
        self.z = self._z()

    def _centerx(self):
        mean = np.array([np.mean(attr) for attr in self.x.T])
        centerx = self.x - mean
        return centerx

    def _cov(self):
        total = np.shape(self.centerX)[0]
        cov = np.dot(self.centerX.T, self.centerX) / (total - 1)
        return cov

    def _u(self):
        a, b = np.linalg.eig(self.c)
        index = np.argsort(-1 * a)
        # b[:, 0: self.k] 返回的特征值不一定有序，排序后依次取前k列特征向量
        # 取完后需转置回特征向量格式
        ut = [b[:, index[i]] for i in range(self.k)]
        print('特征向量:\n', b)
        print('切分方式:\n', ut)
        u = np.transpose(ut)
        print('%d阶降维转换矩阵U:\n' % self.k, u)
        return u

    def _z(self):
        z = np.dot(self.x, self.u)
        return z


if __name__ == '__main__':
    x = np.array([[61, 49, 36, 64, 76, 66, 20, 66, 29, 18],
                  [27, 15, 52, 32, 51, 70, 46, 63, 7, 1],
                  [41, 23, 70, 78, 24, 37, 94, 53, 76, 20],
                  [57, 76, 70, 44, 20, 73, 23, 16, 37, 22],
                  [23, 84, 67, 47, 11, 97, 9, 74, 94, 96],
                  [44, 73, 3, 27, 85, 46, 10, 83, 73, 49],
                  [97, 44, 53, 6, 14, 73, 35, 10, 59, 54],
                  [37, 72, 96, 44, 72, 15, 90, 50, 70, 72]])
    k = np.shape(x)[1] - 1
    res = CPCA(x, k)
