import numpy as np

def pca(X, n_components):
    # 1. 去中心化矩阵
    centrx = X - X.mean(axis=0)
    # 2. 协方差矩阵
    C = np.cov(centrx.T)
    # 3. 特征值和特征向量
    a, b = np.linalg.eig(C)
    # 4. 索引, 排序, 按n_components拼接
    idx = np.argsort(-a)
    U = b[:, idx[:n_components]]
    # 5. 降维
    return np.dot(X, U)

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
    _ = pca(X, k)
    print(_)

if __name__ == "__main__":
    test()