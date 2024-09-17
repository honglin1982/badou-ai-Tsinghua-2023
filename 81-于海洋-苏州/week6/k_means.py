#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/6/7
@author: 81-于海洋

"""
import math
import random

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from typing import List

from config import Config
from utils import ImgPreview


def km_2_img(src, data, k):
    """
    参数：
    data：需要进行聚类的数据，通常是一个N维的浮点型数据矩阵，每行代表一个数据样本。
    K：聚类的簇数。
    criteria：迭代停止的准则。它是一个包含迭代次数（max_iter）和聚类中心变化阈值（epsilon）的元组。
    attempts：重复聚类的次数，算法会返回最优结果。
    flags：额外的标志控制算法的行为，通常设置为cv2.KMEANS_RANDOM_CENTERS以使用随机初始化的聚类中心。

    返回值：
    compactness：返回的紧密度（compactness），表示数据点到其对应簇中心的总平方距离之和，紧密度越小表示聚类效果越好。
    labels：每个数据点的标签，表示其所属的簇。
    centers：每个簇的中心点。
    """
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

    # 图像转换回uint8二维类型
    uint8_c = np.uint8(centers)
    res = uint8_c[labels.flatten()]
    dst = res.reshape((src.shape))
    preview.add(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB), "K-Means:{}".format(k))


def km_2_array(ax, data, kk):
    """
    KMeans 用来分类数组并展示
    :return:
    """
    array = data
    # KMeans
    k = KMeans(n_clusters=kk)
    # 返回一个一维数组 数值大小表示 当前属于几个分类
    pred = k.fit_predict(array)

    print("pred1:", pred)
    # 把array分别转换为 x 和 y 的一维数组
    x = [n[0] for n in array]
    y = [n[1] for n in array]
    ax.scatter(x, y, c=pred, marker='x')


def k_means(ax, data, k, max_iterations=100):
    """
    依据算法原理实现
    """
    # 随机选择K个中心点
    centers = data[np.random.choice(len(data), k, replace=False)]
    pred = []
    for _ in range(max_iterations):
        # 计算每个样本与中心点的距离
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        # 分配样本到最近的中心点
        pred = np.argmin(distances, axis=1)
        # 更新中心点为各个簇的均值
        new_centers = np.array([data[pred == i].mean(axis=0) for i in range(k)])
        # 如果中心点不再变化，停止迭代
        if np.all(centers == new_centers):
            break

        centers = new_centers

        # 把array分别转换为 x 和 y 的一维数组
    print("pred2:", pred)
    x = [n[0] for n in data]
    y = [n[1] for n in data]
    ax.scatter(x, y, c=pred, marker='x')


if __name__ == '__main__':
    # ---- array ---- #
    fig, (ax1, ax2) = plt.subplots(2)
    src_data = np.random.randint(0, 10, size=[30, 2])
    km_2_array(ax1, src_data, 3)
    k_means(ax2, src_data, 3)
    plt.tight_layout()
    plt.show()

    # ---- image ---- #
    preview = ImgPreview(18, 18, 2, 2)
    #
    # cv2 的参数，在 km_2_img 有进一步的解释
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #
    img = cv2.imread(Config.LENNA)
    preview.add(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Origin")

    # img 数组 由 3维->1维
    scr_data = img.reshape((-1, 3))
    scr_data = np.float32(scr_data)
    km_2_img(img, scr_data, 2)
    km_2_img(img, scr_data, 4)
    km_2_img(img, scr_data, 6)
    preview.show()

