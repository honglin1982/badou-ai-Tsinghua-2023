import numpy as np
from sklearn.decomposition import PCA

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

pca = PCA(n_components=2) # n_components降低到的维度
X_new = pca.fit_transform(X) # 训练并且返回训练集降维的数据
print(pca.explained_variance_ratio_) # 输出贡献率
print(X_new)