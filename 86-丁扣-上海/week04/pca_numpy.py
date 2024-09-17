import numpy as np


class PCA(object):

    def __init__(self, n_components: int):
        self.covariance = None
        self.components_ = None
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray):
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print(eig_vals)
        print(eig_vectors)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        print(idx)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)


if __name__ == '__main__':
    # 调用
    pca = PCA(n_components=2)
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1],
                  [3, 5, 83, 2]])  # 导入数据，维度为4
    newX = pca.fit_transform(X)
    print(newX)  # 输出降维后的数据
    '''
    [[  7.96504337  -4.12166867]
     [ -0.43650137  -2.07052079]
     [-13.63653266  -1.86686164]
     [-22.28361821   2.32219188]
     [  3.47849303   3.95193502]
     [ 24.91311585   1.78492421]]
    '''

