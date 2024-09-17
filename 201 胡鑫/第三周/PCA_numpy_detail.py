import numpy as np

class PCA:
    def __init__(self, X, k):
        self.X = X            # 输入样本矩阵
        self.k = k            # 需要将样本矩阵降低到的指定维度
        self.centrx = []      # 样本的去中心化矩阵
        self.C = []           # 通过样本的去中心化矩阵得到的协方差矩阵
        self.U = []           # 样本的降维转换矩阵
        self.Z = []           # 输出样本的降维矩阵

        self.centrx = self._getcentrx()
        self.C = self._getC()
        self.U = self._getU()
        self.Z = self._getZ()

    def _getcentrx(self):
        '''求样本的去中心化矩阵'''
        print('样本矩阵: \n', self.X)
        # 1. 求样本的特征均值
        mean = [np.mean(i) for i in self.X.T] # 转置后此式求的就是原来每列的均值
        # 2. X - mean就为去中心化矩阵(广播机制)
        centrx = self.X - mean
        print('样本的特征均值: \n', mean)
        print('样本的去中心化矩阵: \n', centrx)
        return centrx
    
    def _getC(self):
        '''求样本的协方差矩阵, 利用去中心化矩阵'''
        # 由于已经去中心化, 所以均值为0, 带入公式
        m = self.centrx.shape[0] - 1  # 多数时候无法知晓样本的总体均值, 使用无偏估计
        C = np.dot(self.centrx.T, self.centrx) / m
        print('样本的协方差矩阵: \n', C)
        return C

    def _getU(self):
        '''求样本的降维转换矩阵, 其实就是按权重和k值拼接特征向量'''
        # 1. 利用协方差矩阵求得特征值和特征向量
        a, b = np.linalg.eig(self.C)
        print('特征值: \n', a)
        print('特征向量: \n', b)
        # 2. 按照权重拼接, 首先给特征值排序, 得到a的倒序索引
        idx = np.argsort(-a)
        # 3. 按a的倒序索引和k值重新拼接特征向量b得到降维转换矩阵U
        U = b[:, idx[:self.k]]   # 冒号表示所有行都需要, 后面表示按照idx顺序需要前k列
        print('降维转换矩阵: \n', U)
        return U
    
    def _getZ(self):
        '''求X的降维转换矩阵'''
        # 1. Z = XU
        Z = np.dot(self.X, self.U)
        print('样本X的降维矩阵: \n', Z)
        print('X.shape: ', self.X.shape)
        print('U.shape: ', self.U.shape)
        print('Z.shape: ', Z.shape)
        return Z
    
def test():
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
    k = X.shape[1] - 1
    pca = PCA(X, k)    
if __name__ == "__main__":
    test()
