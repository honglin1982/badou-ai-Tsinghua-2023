import random

import numpy as np

class PCA(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []
        self.centrX = self.centralized()
        self.C = self.cov()
        self.U = self._U()
        self.Z = self._Z()
    #中心化
    def centralized(self):
        XT = [[row[i] for row in self.X] for i in range(len(X[0]))]
        mean = np.array([np.mean(i) for i in XT])
        centrX = self.X - mean
        return centrX
    #求协方差矩阵
    def cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('样本数据X经过PCA的降维矩阵：\n', Z)
        return Z

if __name__ == '__main__':
    X = [[np.random.randint(100) for i in range(6)] for j in range(10)]
    print("样本数据X:",X)
    K = int(input('希望将成几维：'))
    pca = PCA(X, K)