# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt


def normalization(data_set):
    """
    标准化，将数据映射到[0,1]区间上
    :param data_set: 数据集
    :return: 标准化后的数据
    """
    # y = (x-min)/(max-min)
    arr = []
    min_x = min(data_set)
    max_x = max(data_set)
    for data in data_set:
        arr.append((data - min_x)/(max_x - min_x))
    return arr


def normalization2(data_set):
    """
    标准化，将数据映射到[-1,1]区间上
    :param data_set: 数据集
    :return: 标准化后的数据
    """
    # y = (x-mean)/(max-min)
    mean = np.mean(data_set)
    arr = []
    min_x = min(data_set)
    max_x = max(data_set)
    for data in data_set:
        arr.append((data - mean)/(max_x - min_x))
    return arr


def z_score(data_set):
    """
    零均值归一化
    经过处理后的数据均值为0，标准差为1（正态分布）
    :param data_set: 数据集
    :return:
    """
    # y = (x-μ)/σ
    arr = []
    # 计算均值
    miu = np.mean(data_set)
    # 求和
    sumData = 0
    for data in data_set:
        sumData += (data - miu)**2
    # 计算标准差
    sigma = sumData / len(data_set)
    for data in data_set:
        arr.append((data - miu) / sigma)
    return arr


if __name__ == '__main__':
    # 数据集
    data_set = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
    # 存储每个数的数量
    count_arr = []
    for i in data_set:
        count_arr.append(data_set.count(i))

    result1 = normalization(data_set)
    result2 = normalization2(data_set)
    result3 = z_score(data_set)
    # 展示原数据集
    plt.plot(data_set, count_arr)
    plt.plot(result1, count_arr)
    plt.plot(result2, count_arr)
    plt.plot(result3, count_arr)
    plt.show()
