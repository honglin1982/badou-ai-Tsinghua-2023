# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/6/28
@author: 81-于海洋
"""

import numpy as np
import matplotlib.pyplot as plt


# 归一化的两种方式
def Normalization1(x):
    """归一化（0~1）"""
    '''x_=(x−x_min)/(x_max−x_min)'''
    max_num = max(x)
    min_num = min(x)
    print("max:", max_num, ",min:", min_num)
    return [(float(_x) - min_num) / float(max_num - min_num) for _x in x]


def Normalization2(x):
    """归一化（-1~1）"""
    '''x_=(x−x_mean)/(x_max−x_min)'''
    mean = np.mean(x)
    return [(float(i) - mean) / (max(x) - min(x)) for i in x]


# 标准化
def z_score(x):
    """
    公式：
    x∗=(x−μ)/σ
    μ：样本的均值
    σ：样本的标准差
    """
    # μ：样本的均值 计算
    x_mean = np.mean(x)
    # σ：样本的标准差 计算
    s2 = sum([(i - x_mean) * (i - x_mean) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


if __name__ == '__main__':
    src_list = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11,
                11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

    src_num_count = []
    for idx in src_list:
        c = src_list.count(idx)
        src_num_count.append(c)

    print("src_num_count:", src_num_count)
    n_1 = Normalization1(src_list)
    n_2 = Normalization2(src_list)
    z = z_score(src_list)
    '''
    蓝线为原始数据，橙线为z
    '''
    plt.plot(src_list, src_num_count)
    plt.plot(n_1, src_num_count)
    plt.plot(n_2, src_num_count)
    plt.plot(z, src_num_count)
    plt.show()
