# -*- coding: utf-8 -*-
import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cluster_func(centers, dataSet):
    cluster = {}
    cluster_data = {}
    for i in range(len(centers)):
        cluster[i] = []
        cluster_data[i] = []
    # 3. 分别计算每个点到每个质心之间的距离，并将每个点划分到离最近质心的小组。
    for idx in range(len(dataSet)):
        data = dataSet[idx]
        distance_arr = []
        for i in range(len(centers)):
            distance_arr.append(math.sqrt(np.square(data - centers[i]).sum()))
        cluster_idx = distance_arr.index(min(distance_arr))
        cluster[cluster_idx].append(idx)
        cluster_data[cluster_idx].append(data)
    # 4. 当每个质心都聚集了一些点后，重新定义算法选出新的质心。（对于每个簇，计算其均值，即得到新的k个质心点）
    new_centers = []
    for i in range(len(cluster_data)):
        new_centers.append(np.average(cluster_data[i], axis=0))
    change = np.array(new_centers) - np.array(centers)
    return change, new_centers, cluster


if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png', 0)
    # 1. 确定K值，即将数据集聚集成K个类簇或小组
    K = 12
    # 2. 从数据集中随机选择K个数据点作为质心（Centroid）或数据中心。
    height, width = img.shape
    dataSet = img.reshape((int((height * width) / 2), 2))
    dataSet = np.float32(dataSet)
    centers = []
    for i in range(K):
        centers.append(dataSet[random.randint(0, dataSet.shape[0])])
    print(centers)
    num = 0
    change, new_centers, cluster = cluster_func(centers, dataSet)
    # 5. 迭代执行第三步到第四步，直到迭代终止条件满足为止
    while np.any(change != 0) and num < 20:
        num += 1
        centers = new_centers
        change, new_centers, cluster = cluster_func(new_centers, dataSet)
    print(num, change, centers, new_centers)
    # 修改原图的值为各个簇心的值
    for i in range(len(centers)):
        for d in cluster[i]:
            dataSet[d] = centers[i]
    # 还原图像的形状
    result = dataSet.reshape(img.shape)
    result = np.uint8(result)
    # 显示图像
    cv2.imshow("1", result)
    cv2.waitKey(0)
