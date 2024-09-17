# -*- coding: utf-8 -*-
# @Time    : 2023/6/9 9:56
# @Author  : zgh
# @FileName: kmean_img.py
# @Software: PyCharm


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('lane.jpg',0)
w, h = img.shape[:]

data = img.reshape((w * h, 1))
data = np.float32(data)

criteria = (cv.TERM_CRITERIA_EPS +
            cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 随机中心
flags = cv.KMEANS_RANDOM_CENTERS

"""
data：输入的数据，需要是 numpy 数组或矩阵形式。
4：要聚类的簇数，即聚类后得到的簇的数量。
None：簇标记，如果给出，则此函数将使用这些标记开始聚类过程，否则随机选择一些标记作为起点。
criteria：定义停止条件。
10：重复运行 k-means 算法的次数，以获得更好的结果。
flags：定义 k-means 的行为模式和细节。

compactness：float 类型的值，表示用于评估当前聚类的紧密度指标。该指标越小，表示聚类效果越好。在 k-means 算法的过程中，该指标会逐渐减小。
labels：与输入数据大小相同的 int 类型 numpy 数组，其中包含每个数据点所属的簇的标签。
centers：一个由聚类中心构成的 numpy 数组，其形状为 (num_clusters, num_features)。每个聚类中心都是一个特征向量，它代表该簇中所有数据点的平均值。
"""
compactness, labels, centers = cv.kmeans(data, 4, None, criteria, 10, flags)

plt.rcParams['font.sans-serif'] = ['SimHei']

result = labels.reshape((w, h))

titles = [u'原始图像', u'聚类图像']
images = [img, result]
for i in range(2):
	plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
	plt.title(titles[i])
	plt.xticks([]), plt.yticks([])
plt.show()
