'''
pca
'''

import numpy as np
from sklearn.decomposition import PCA

def pca_detail(x, k):
    # 中心化
    x -= np.mean(x, axis=0)
    # 协方差矩阵
    cov = np.cov(x.T)
    # 求特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(cov)
    # 排序
    eigValIndice = np.argsort(eig_val)
    n_eigValIndice = eigValIndice[-1:-(k + 1):-1]
    remain_vec = eig_vec[:, n_eigValIndice]

    return np.dot(x, remain_vec)


def pca_sklearn(x, k):
    pca = PCA(n_components=k)
    pca.fit(x)
    return pca.transform(x)




if __name__ == '__main__':
    x = np.array([[10, 15, 20],
                  [15, 46, 13.0]])

    print(pca_detail(x, 2))
    print(pca_sklearn(x, 2))
