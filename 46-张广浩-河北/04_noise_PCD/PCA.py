import numpy as np

from sklearn.decomposition import PCA


def PCA_detail(data, k):
	data_move_mean = []
	cov_m = []
	trans_m = []
	result = []
	# 这里注意，原始data每行是一个样本，每列是其特征
	"""
	如：
		a b c
	1
	2
	3
	4
	此时求均值得到的是每个样本的所有均值（4） 如1的 a+b+c // 3 
	转置后：
		1 2 3 4
	a
	b
	c
	此时求均值得到的是所有样本每个属性的均值(3) 如a的 1+2+3+4 // 4
	"""
	# 1,去均值 data_move_mean.shape = [10, 3]
	mean = np.array([np.mean(x_col) for x_col in data.T])
	data_move_mean = data - mean
	print("去均值：data_move_mean.shape:", data_move_mean.shape)

	# 2，求协方差 cov_m.shape = [3, 10] * [10, 3] = [3, 3]
	# 公式 m = (M.T * M)/(n-1)
	n = data_move_mean.shape[0]
	cov_m = np.dot(data_move_mean.T, data_move_mean) / (n - 1)
	print("协方差cov_m.shape:", cov_m.shape)

	# 3,求特征值和特征向量,得到转换矩阵  trans_m.shape = [3, k] k<=3
	feature_value, feature_voter = np.linalg.eig(cov_m)
	print("feature_value.shape:", feature_value.shape, "feature_voter.shape:", feature_voter.shape)
	# 从大到小排序得到索引，添加负号成为从大到小
	ind = np.argsort(-1 * feature_value)
	# 选择最大两个特征值对应的特征向量
	trans_m_T = [feature_voter[:, ind[i]] for i in range(k)]
	trans_m = np.transpose(trans_m_T)
	print("转换矩阵trans_m.shape:", trans_m.shape)

	# 4,原始数据于转化矩阵相乘得到降维后数据
	result = np.dot(data, trans_m)
	print("降维后数据result.shape:", result.shape)

	return result


def PCA_auto(data, k):
	pca = PCA(n_components=k)
	pca.fit(data)  # 训练
	result = pca.fit_transform(data)
	return result


if __name__ == '__main__':
	'10样本3特征的样本集, 行为样例，列为特征维度'
	X = np.array([[10, 15, 29],
	              [15, 46, 13],
	              [23, 21, 30],
	              [11, 9, 35],
	              [42, 45, 11],
	              [9, 48, 5],
	              [11, 21, 14],
	              [8, 5, 15],
	              [11, 12, 21],
	              [21, 20, 25]])
	K = np.shape(X)[1] - 1
	print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
	print("PCA接口版本：")
	pca_result = PCA_auto(X, K)
	print('降维之后的结果(10行2列，10个样例，每个样例2个特征):\n', pca_result)
	print(pca_result.shape)
	print("#" * 30)
	print("PCA手动实现版本：")
	pca_result = PCA_detail(X, K)
	print('降维之后的结果(10行2列，10个样例，每个样例2个特征):\n', pca_result)
