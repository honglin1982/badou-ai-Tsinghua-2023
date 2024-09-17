import numpy as np


def PCA_detail(X, K):
    print("样本矩阵X:\n", X, X.shape)

    # 矩阵中心化
    mean = np.array([np.mean(attr) for attr in X.T])    # 样本集的特征均值
    print("样本集的特征均值：\n", mean)
    centerX = X - mean  # 矩阵中心化
    print("样本集的中心化矩阵：\n", centerX, centerX.shape)

    # 协方差矩阵D = Z.T * Z / (m - 1)
    m = centerX.shape[0]
    C = np.dot(centerX.T, centerX) / (m - 1)
    print("样本矩阵X的协方差矩阵C:\n", C, C.shape)

    # 获取协方差矩阵C的特征值和特征向量，a表示特征值，b表示对应特征向量
    a, b = np.linalg.eig(C)
    print("协方差矩阵C的特征值：\n", a)
    print("协方差矩阵C的特征向量：\n", b, b.shape)

    # 获取特征值降序的索引序列
    idx = np.argsort(-a)

    # 构建K阶降维的降维转换矩阵，即按上一步的降序索引组合特征向量，形成矩阵
    UT = [b[:, idx[i]] for i in range(K)]
    U = np.transpose(UT)
    print(f"{K}阶降维转换矩阵U：\n", U, U.shape)

    # 降维矩阵Z=X*U
    Z = np.dot(X, U)
    print("样本矩阵X的降维矩阵Z:\n", Z, Z.shape)

    return Z


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
pca = PCA_detail(X, K)
