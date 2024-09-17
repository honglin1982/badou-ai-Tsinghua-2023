import numpy as np

class PCAD(object):

    def __init__(self, X, K):

        self.X = X           #输入样本矩阵
        self.K = K           #输入要降低的维度
        self.central = []    #中心化矩阵
        self.C = []          #协方差矩阵
        self.U = []          #降维转换矩阵
        self.Z = []          #降维矩阵

        self.central = self.Centralized()
        self.C = self.cov()
        self.U = self.Transform()
        self.Z = self.XU()

    def Centralized(self):

        print("输入样本矩阵:\n", self.X);
        mean = np.array([np.mean(col) for col in self.X.T])
        print("特征均值:\n", mean)
        central = self.X - mean
        print("中心化后矩阵:\n", central)
        return central

    def cov(self):

        m = self.X.shape[0]
        cov = np.dot(self.central.T, self.central) / m
        print("协方差矩阵:\n", cov)
        return cov

    def Transform(self):

        eigVals, eigVectors = np.linalg.eig(self.C)
        print("样本特征值:\n", eigVals)
        print("样本特征向量:\n", eigVectors)
        idx = np.argsort(eigVals * -1)
        U = np.transpose([eigVectors[:,idx[i]] for i in range(self.K)])
        print("%d阶降维转换矩阵:\n"%self.K, U)
        return U

    def XU(self):

        Z = np.dot(self.X, self.U)
        print("X shape", np.shape(self.X))
        print("U shape", np.shape(self.U))
        print("Z shape", np.shape(Z))
        print("X样本集的%d阶降维矩阵:\n"%self.K, Z)
        return Z

def fit_transform(X, components):

     m = X.shape[0]
     central = X - X.mean(axis=0)
     cov = np.dot(central.T, central) / m
     eigVals, eigVectors = np.linalg.eig(cov)
     idx = np.argsort(eigVals * -1)
     U = np.transpose([eigVectors[:,idx[i]] for i in range(components)])
     Z = np.dot(X, U)
     print("X样本集的%d阶降维矩阵:\n" %components, Z)

if __name__ == '__main__':

     X = np.random.randint(-100, 100,(10, 3))
     PCAD(X, X.shape[1] - 1)

     fit_transform(X,  X.shape[1] - 1)