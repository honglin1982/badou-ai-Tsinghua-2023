import numpy as np

class CPCA(object):

    def __init__(self,X,K) :
        
        self.X = X  #样本矩阵X
        self.K = K  #K阶降维矩阵的K值
        self.centerX = [] #样本矩阵X的均值中心化矩阵
        self.C = []  #样本集的协方差矩阵C
        self.U = []  #样本矩阵X的降维转换矩阵
        self.Z = []  #样本矩阵X的降维矩阵

        self.centerX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        """样本矩阵X的中心化"""
        print('中心化样本矩阵X：\n',self.X)
        centerX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) #样本集的特征均值
        print('样本集的特征均值：\n',mean)
        centerX = self.X - mean  #样本集的中心化
        print('样本集中心化矩阵centerX：\n',centerX)
        return centerX
    def _cov(self):
        """求样本的协方差矩阵C"""
        #样本总数（矩阵的行为样本数，列为特征数
        ns = self.X.shape[0]
        #样本集的协方差矩阵C
        C = np.dot(self.centerX.T,self.centerX)/(ns-1)
        print('样本矩阵的协方差矩阵C：\n',C)
        return C
    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵C的特征值和特征向量
        a , b = np.linalg.eig(self.C) #特征值赋给a，特征向量赋给b
        print('样本集的协方差矩阵C的特征值a：\n',a)
        print('样本集的协方差矩阵C的特征向量b：\n',b)
        #给出特征值Top K的降序序列
        ind = np.argsort(-1*a)
        #构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K) ]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X,self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    
    K = np.shape(X)[1]-1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)





