#手动实现pca类对样本矩阵实现降维

import numpy as np

class PCA(object):
    def __init__(self,sampleMatrix,descendingDimension):
        #样本矩阵X
        self.smX=sampleMatrix
        #降维矩阵的阶数
        self.ddK=descendingDimension
        #中心化后的样本矩阵
        self.centralX=[]
        #样本矩阵的协方差矩阵
        self.covX=[]
        #降维后的矩阵
        self.ddY=[]
        #样本的降维转换矩阵
        self.ddT=[]

        #采用内部函数执行pca
        self.centralX=self.centralize()
        self.covX=self.covariation()
        self.ddT=self.dimReductionTransMatrix()
        self.ddY=self.dimReductionMatrix()
    def centralize(self):
        #矩阵的中心化
        centralX=[]
        mean=np.array([np.mean(attr) for attr in self.smX.T])
        centralX=self.smX-mean
        return centralX
    def covariation(self):
        #求样本矩阵的协方差矩阵
        numOfSample=np.shape(self.smX)[0]
        covX=np.dot(self.centralX.T,self.centralX)/(numOfSample-1)
        return covX
    def dimReductionTransMatrix(self):
        #先求协方差矩阵的特征值和特征向量
        eigValue,eigVector=np.linalg.eig(self.covX)
        #特征值降序排列的索引序列
        index=np.argsort(-1*eigValue)
        #构建k阶降维矩阵
        ddTT=[eigVector[:,index[i]] for i in range(self.ddK)]
        ddT=np.transpose(ddTT)
        return ddT
    def dimReductionMatrix(self):
        ddY=np.dot(self.smX,self.ddT)
        print(ddY)
        return ddY

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X,K)