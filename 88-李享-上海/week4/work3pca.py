# pca输入行列数目，读取excel文件，降维到目标维数
# 洗数据，求平均值，减去均值 求协方差
# n个特征组成n阶的矩阵每一项对于两组数据的协方差
# 对目标矩阵求特征向量和特征值 矩阵相乘前行后列获取全特征矩阵
# 按特征值从大到小取对于特征向量组成新的特征矩阵。
# 将原数据与新特征矩阵相乘组成新的数据集，带有原始数据的特征。
import numpy as np

def pca(X, num_components):
    # 数据预处理：计算均值并减去均值
    mean_X = np.mean(X, axis=0)
    X_normalized = X - mean_X

    # 计算协方差矩阵
    covariance_matrix = np.cov(X_normalized, rowvar=False)

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 特征值排序与选择
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, sorted_indices[:num_components]]

    # 特征向量投影
    transformed_X = np.dot(X_normalized, selected_eigenvectors)

    return transformed_X

# 示例数据
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用 PCA 进行降维，指定保留两个主成分
transformed_data = pca(data, 2)

print(transformed_data)
