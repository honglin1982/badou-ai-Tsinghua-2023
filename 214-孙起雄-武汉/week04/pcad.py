# 实现PCA（detail）

import numpy as np


def pca(X, K):
    # 1. 对样本矩阵进行中心化，即减去每个特征的均值
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. 计算样本矩阵的协方差矩阵
    cov_matrix = np.cov(X_centered.T)

    # 3. 对协方差矩阵进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 对特征值进行排序，选择前K个最大特征值对应的特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_eigenvectors = eigenvectors[:, sorted_indices[:K]]

    # 5. 将样本矩阵投影到选取的特征向量构成的新空间
    Z = np.dot(X_centered, top_k_eigenvectors)

    return Z

# 示例用法：
# 假设有一个样本矩阵X，形状为(10, 3)，其中m是样本数，n是特征数

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

# 指定降维后的维度K
K = 2
# 调用PCA函数计算降维后的特征矩阵Z
Z = pca(X, K)
# 输出降维后的特征矩阵Z
print(Z)
