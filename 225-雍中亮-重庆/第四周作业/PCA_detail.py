import numpy as np
class PCA(object):
    # X为样本矩阵，K为降维矩阵Z的阶数
    def __init__(self,X,K):
        self.X=X
        self.K=K
        self.centrX=[] #矩阵的中心化
        self.C=[]    #样本集的协方差矩阵C
        self.U=[]    #样本矩阵X的降维转换矩阵
        self.Z=[]    #样本矩阵X的降维矩阵Z

        self.centrX=self._centralized()
        self.C=self._cov()
        self.U=self._U()
        self.Z=self._Z()

    def _centralized(self):
        # 矩阵的中心化
        print("样本矩阵X:\n",self.X)
        centrX=[]
        mean=np.array([np.mean(i) for i in self.X.T])
        print('样本的特征均值:\n',mean)
        centrX=self.X-mean
        print('样本矩阵X的中心化centrX:\n',centrX)
        return centrX

    def _cov(self):
        # 中心矩阵协方差化
        # 先求样本的个数
        ns=np.shape(self.X)[0]
        # 求协方差矩阵
        C=np.dot(self.centrX.T, self.centrX) / (ns-1)
        print('样本矩阵X的协方差矩阵C:\n',C)
        return C

    def _U(self):
        # 求X的降维转换矩阵U
        # 先求特征值和特征向量
        a,b=np.linalg.eig(self.C) #特征值给a，特征向量给b。
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind=np.argsort(-1*a)
        # 给出K阶的降维的转换矩阵U
        UT=[b[:,ind[i]] for i in range(self.K)]
        U=np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K,U)
        return U

    def _Z(self):
        #求最后的降维矩阵Z
        Z=np.dot(self.X,self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

X=np.array([[10, 15, 29],
            [15, 46, 13],
            [23, 21, 30],
            [11, 9,  35],
            [42, 45, 11],
            [9,  48, 5],
            [11, 21, 14],
            [8,  5,  15],
            [11, 12, 21],
            [21, 20, 25]])
K=2
pca=PCA(X,K)



