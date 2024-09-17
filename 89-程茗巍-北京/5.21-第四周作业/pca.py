import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

class PCA(object):
    def __init__(self,points_np, K):
        self.points_np = points_np
        self.K = K
        self.centr_points = []
        self.mean = []
        self.covariance_points = []
        self.U = []
        self.sort_eigenvectors = []
        self.Z = []
        self.centr_points, self.mean = self.centralized()
        self.covariance_points = self.cov()
        self.U, self.sort_eigenvectors = self._U()
        self.Z = self._Z()

    def centralized(self):
        print('矩阵样本：\n', self.points_np)
        mean = np.mean(self.points_np, axis=0)
        print('样本集的特征均值：\n', mean)
        centr_points = self.points_np - mean
        print('样本矩阵X的中心化:\n',  centr_points)
        return centr_points, mean

    def cov(self):
        #print(self.centr_points)
        #计算协方差矩阵
        covariance_points = np.cov(self.centr_points.T)
        print('样本矩阵X的协方差矩阵C:\n', covariance_points)
        return covariance_points

    def _U(self):
        # 进行特征分解
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_points)
        print('样本集的协方差矩阵C的特征值:\n', eigenvalues)
        print('样本集的协方差矩阵C的特征向量:\n', eigenvectors)
        # 将特征值进行排序
        sort_indices = np.argsort(eigenvalues)[::-1]
        sort_eigenvectors = eigenvectors[:, sort_indices]
        print('样本集的协方差矩阵C的特征值序列为:\n', sort_indices)
        print('样本集的协方差矩阵C的特征值向量排序后矩阵为:\n', sort_eigenvectors)
        principal_components = sort_eigenvectors[:, :self.K]
        print('%d阶降维转换矩阵U:\n' % self.K, principal_components)
        U = principal_components
        return U, sort_eigenvectors

    def _Z(self):
        #将原始点云投影到新的基向量上
        Z = np.dot(self.points_np, self.U)
        print('points_np shape:', np.shape(self.points_np))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == "__main__":
    # 加载点云数据
    file_name = "cup_0001.npy"
    points_np = np.load(file_name)[:,0:3]
    K = np.shape(points_np)[1] - 1
    pca = PCA(points_np, K)

    #画出原点云
    fig = plot.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(points_np[:,0],points_np[:,1],points_np[:,2])
    #画出降维方向
    ax.plot([pca.mean[0], pca.mean[0] + pca.sort_eigenvectors[0,0]],
            [pca.mean[1], pca.mean[1] + pca.sort_eigenvectors[1,0]],
            [pca.mean[2], pca.mean[2] + pca.sort_eigenvectors[2,0]], color='r', linewidth=3)
    ax.plot([pca.mean[0], pca.mean[0] + pca.sort_eigenvectors[0,1]],
            [pca.mean[1], pca.mean[1] + pca.sort_eigenvectors[1,1]],
            [pca.mean[2], pca.mean[2] + pca.sort_eigenvectors[2,1]], color='g', linewidth=3)
    plot.show()
#scikit-learn 实现
# pca = PCA(n_components)
# transform_points1 = pca.fit_transform(points_np)
# print('transform_points1 = ',transform_points1)