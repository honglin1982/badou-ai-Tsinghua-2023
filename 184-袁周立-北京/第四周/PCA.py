import numpy as np
from sklearn.datasets.base import load_iris
from sklearn.decomposition import PCA


def DIY_PCA(data, k):
    data2 = np.copy(data)
    # 中心化
    data2 = data2 - np.mean(data2, axis=0)
    # 求协方差矩阵
    covariance_matrix = np.dot(data2.T, data2) / data2.shape[0]
    # 求协方差矩阵的特征值，特征向量
    eigvalues, eigvectors = np.linalg.eig(covariance_matrix)
    # 取前k个特征值对应的特征向量组成矩阵
    matrix_w = eigvectors[:, np.argsort(-eigvalues)[:k]]
    return np.dot(data, matrix_w)


x,y=load_iris(return_X_y=True)

dst1 = DIY_PCA(x, 2)

sk_pca = PCA(2)
dst2 = sk_pca.fit_transform(x)

print(dst1.shape, dst2.shape)

for i in range(dst1.shape[0]):  # 得到的结果不一样，sklearn得到的特征向量有的是numpy计算的特征向量的相反数，似乎与svd有关
    print(dst1[i], dst2[i])

