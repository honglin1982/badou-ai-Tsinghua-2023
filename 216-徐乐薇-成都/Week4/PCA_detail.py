import numpy as np

class ClassPCA(object):

    def __init__(self, X, K):
        #X为训练样本矩阵(m行样例，n列特征)，K为X降维矩阵
        self.X = X #样本矩阵X
        self.K = K #K阶降维矩阵的阶数-X要特征降维成k阶
        self.centrX = []
        self.C = [] #样本集的协方差矩阵C
        self.U = [] #样本矩阵X的降维转换矩阵
        self.Z = [] #样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU

    #样本矩阵X的协方差矩阵C
    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        #样本集的特征均值
        mean = np.array([np.mean(elem) for elem in self.X.T])
        print('样本集的特征均值:\n', mean)
        # 样本集中化
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    #求样本矩阵X的协方差矩阵C
    def _cov(self):
        samplenum = np.shape(self.centrX)[0] #样本集样例总数
        C = np.dot(self.centrX.T, self.centrX)/(samplenum - 1) #样本矩阵的协方差矩阵C
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    #求X的降维转换矩阵U，shape=(n,k)，其中n：X的特征维度总数，k:降维矩阵的特征维度
    def _U(self):
        #先求X的协方差矩阵C的特征值和特征向量
        a,b = np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        #特征值降序的topK的索引序列
        index = np.argsort(-1*a)
        #构建降维后 K阶降维转换矩阵U
        UT = [b[:,index[i]] for i in range(self.K)]
        U = np.transpose(UT) # np.transpose用于调换数组行列值的索引值，类似于矩阵的转置
        print('%d阶降维转换矩阵U:\n'%self.K, U)
        return

    #Z=XU求降维矩阵Z，shape=(m,k),m是样本总数，k是降维矩阵中特征维度
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

#8样本5特征的样本集，行-样例，列-特征维度
if __name__ == '__main__':
    X = np.array([
            [8, 15, 5, 6, 6],
            [125, 6, 42, 34, 1],
            [11, 35, 8, 48, 20],
            [15, 21, 25, 31, 4],
            [4, 12, 144, 52, 6],
            [125, 4, 72, 31, 6],
            [13, 11, 1, 34, 27],
            [30, 2, 28, 27, 4]])
    K = np.shape(X)[1] - 1
    print('样本集-8行5列:\n', X)
    pca = ClassPCA(X, K)
