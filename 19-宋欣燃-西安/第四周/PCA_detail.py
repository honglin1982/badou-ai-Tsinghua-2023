"""PCA算法：样本矩阵X的K阶降维矩阵Z"""

import numpy as np
class PCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centerX = [] #矩阵X的中心化
        self.C = [] #协方差矩阵
        self.A = [] #X的降维转换矩阵
        self.Z = [] #X的降维矩阵

        self.centerX = self.centralized()
        self.C = self.cov()
        self.A = self.calA()
        self.Z = self.Z()


    # 样本矩阵X的中心化
    def centralized(self):
        print('样本矩阵X：\n',self.X)
        centerX = []
        mean = np.array([np.mean(i) for i in self.X.T])
        print('样本集的特征均值：\n', mean)
        centerX = self.X-mean
        print('样本矩阵X的中心化centerX：\n', centerX)
        return centerX

    # 求X的协方差矩阵C
    def cov(self):
        num = np.shape(self.centerX)[0]
        C = np.dot(self.centerX.T, self.centerX)/(num-1)
        print('协方差矩阵C：\n', C)
        return C

    # 求X的降维转换矩阵A
    def calA(self):
        a, b = np.linalg.eig(self.C)
        print('协方差矩阵C的特征值为：\n', a)
        print('协方差矩阵C的特征向量为：\n', b)
        index = np.argsort(-1*a) # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引号)
        AT = [b[:,index[i]] for i in range(self.K)]
        A = np.transpose(AT)
        print("%d阶降维转换矩阵A：\n"%self.K, A)
        return A

    # 求降维矩阵Z
    def Z(self):
        Z=np.dot(self.X, self.A)
        print('X shape:', np.shape(self.X))
        print('A shape:', np.shape(self.A))
        print('Z shape:', np.shape(Z))
        print('降维矩阵Z：\n', Z)
        return Z

if __name__ == '__main__':
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    K = np.shape(X)[1]-1
    print('样本集(4行3列，4个样例每个样例3个特征):\n', X)
    pca = PCA(X, K)