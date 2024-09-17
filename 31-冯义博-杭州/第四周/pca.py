import numpy as np


def pca(source, k):
    # 零均值化
    avg = np.mean(source, axis=0)
    center_s = source - avg
    total = np.shape(center_s)[0]
    # 计算协方差
    d = np.dot(center_s.T, center_s) / total
    # 计算特征值和特征向量
    a, b = np.linalg.eig(d)
    # 大小倒排 取下标
    idx = np.argsort(a)[::-1]
    # 取前K列组成降为矩阵
    u = b[:, idx[:k]]
    n_s = np.dot(source, u)
    print("降维矩阵", n_s)


s = np.random.randint(0, 100, size=(10, 3))
print("原矩阵", s)
pca(s, 2)

