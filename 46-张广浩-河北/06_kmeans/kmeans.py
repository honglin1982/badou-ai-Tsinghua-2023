# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 15:22
# @Author  : zgh
# @FileName: kmeans.py
# @Software: PyCharm

import math
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def distance(points, target_points):
	# m 个样本点 , n个聚类中心
	# (m,n)每个样本点对应距离中心的距离
	distances = []
	for x, y in points:
		# 单个样本点对所有距离中心的距离
		dis = []
		for xc, yc in target_points:
			d = math.dist((x, y), (xc, yc))
			dis.append(d)
		distances.append(dis)
	return np.array(distances)


def kmeans(points, k, num_loop=15, dist=np.mean):
	"""
	:param num_loop: 循环次数，num_loop次各个样本聚类结果相同后选择该结果作为最终结果
	:param points: 所有样本点
	:param k: 聚类中心数目
	:param dist: 重新确定聚类中心方法 可以采用
	np.max 选择该簇中最大位置值作为新的中心
	np.min 选择最小值点作为中心
	np.median 选择中位数点作为中心
	np.mean 选择平均值作为中心

	其中最大最小值点每次结果一致，其余方法每次聚类结果可能不同
	:return:
	center:(k,2)各个聚类中心，以及其坐标
	final_center:(m)各个样本点对应的类别
	"""
	assert k >= 1, "聚类中心数目需要大于等于1"
	num_point = points.shape[0]
	# k个聚类中心 (3,2) 3个点每个点2坐标
	center = points[np.random.choice(num_point, size=k, replace=False)]
	# 各个样本点的类别
	final_center = np.zeros((num_point,))
	n = 0
	while True:
		distances = distance(points, center)
		# 取(m,n)各个点到中心的距离，样本点到中心最小的那个中心返回(m)，也就是各个样本的类别
		cur_cluster = np.argmin(distances, axis=1)
		# 如果所有点聚类结果同上次一样，跳出循环
		if (final_center == cur_cluster).all():
			n += 1
			if n == num_loop:
				break
		# 跟新聚类中心
		for i in range(k):
			# 采用np.mean均值的方法，对当前聚类中心对应的样本点取均值，作为新的聚类中心
			center[i] = dist(points[cur_cluster == i], axis=0)
		final_center = cur_cluster

	print(center.shape)
	print(final_center.shape)
	print(center)
	print(final_center)
	return center, final_center


def kmeans_api(data, k):
	# 调用api每次结果都是相同的
	kmean = KMeans(k, init='random')
	print(kmean)
	final_center = kmean.fit_predict(data)
	center = kmean.cluster_centers_
	print(center.shape)
	print(final_center.shape)
	print(center)
	print(final_center)
	return center, final_center


def main():
	# x = np.random.randn(50, 2)
	x = [[0.0888, 0.5885],
	     [0.1399, 0.8291],
	     [0.0747, 0.4974],
	     [0.0983, 0.5772],
	     [0.1276, 0.5703],
	     [0.1671, 0.5835],
	     [0.1306, 0.5276],
	     [0.1061, 0.5523],
	     [0.2446, 0.4007],
	     [0.1670, 0.4770],
	     [0.2485, 0.4313],
	     [0.1227, 0.4909],
	     [0.1240, 0.5668],
	     [0.1461, 0.5113],
	     [0.2315, 0.3788],
	     [0.0494, 0.5590],
	     [0.1107, 0.4799],
	     [0.1121, 0.5735],
	     [0.1007, 0.6318],
	     [0.2567, 0.4326],
	     [0.1956, 0.4280]
	     ]
	data = np.array(x)
	num_cluster = 3
	# 调用自己实现的聚类方法每次结果还不太一样
	# 手动实现Kmeans方式
	#center, final_center = kmeans(data, num_cluster)

	# 调用接口实现方式
	center, final_center = kmeans_api(data, num_cluster)
	x1 = data[:, 0]
	y1 = data[:, 1]
	fig = plt.figure(figsize=(8, 8))
	# c 参数用于指定散点的颜色或颜色映射，cmap 参数用于指定指定颜色映射表（即将数值映射到颜色空间）。
	plt.scatter(x1, y1, c=final_center, cmap='viridis')
	plt.colorbar()
	plt.scatter(center[:, 0], center[:, 1], c='gray', s=200, alpha=0.9)  # 中心点
	plt.show()

"""
两种实现方法差异：
随机初始值的不同：K-means 算法对于初始值的敏感度很高，不同的初始质心可能会导致不同的聚类结果。
当使用 Scikit-learn 的 KMeans 类时，默认情况下，算法会多次随机生成初始化质心，并选择 SSE 最小的结果。
但是，对于手动实现的 K-means 算法来说，初始值通常是随机选择的。因此，两种方法的初始值可能不同，导致聚类结果的差异。

停止条件的不同：K-means 算法的停止条件有多种选择，例如最大迭代次数、收敛阈值等等。
手动实现的算法与 Scikit-learn 的 KMeans 类也可能采用不同的停止条件。这也可能导致不同的聚类结果。

超参数的选择不同：除了需要选择聚类数量 k 以外，K-means 算法还有许多其他的超参数，
例如初始值的方式、距离度量、迭代次数等等。不同的超参数选择可能会导致不同的聚类结果。
"""
if __name__ == '__main__':
	main()
