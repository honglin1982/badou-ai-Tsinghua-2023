#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/6/14
@author: 81-于海洋
"""
import numpy as np
import scipy as sp


def gen_arr(count=10, inp=1, out=1):
    """
    y = k * x + b
    """
    arr_k = 60 * np.random.normal(size=(inp, out))
    arr_b = np.full((count, inp), 3)

    arr_x = 20 * np.random.random((count, inp))
    arr_y = np.dot(arr_x, arr_k) + arr_b  # y = x * k + b
    print("K:", arr_k, "\nB:", arr_b)
    return arr_x, arr_y


def least_squ(src_x, src_y):
    """
    最小二乘法（Least Square Method）
    (Xi,Yi)(i...m)  | h(x) = kx + b
    残差：r = h(Xi) - Yi
    拟合程度，用通俗的话来讲，就是我们的拟合函数h(x)与待求解的函数y之间的相似性。那么 2-范数越小，自然相似性就比较高了。
    r 越小 拟合程度越大
    """
    n = len(src_x)
    total_x = 0
    total_y = 0
    total_x_y = 0
    total_x_x = 0
    for i in range(n):
        total_x = total_x + src_x[i]
        total_y = total_y + src_y[i]
        total_x_x = total_x_x + src_x[i] * src_x[i]
        total_x_y = total_x_y + src_x[i] * src_y[i]

    k = (n * total_x_y - total_x * total_y) / (n * total_x_x - total_x * total_x)
    b = (total_y - k * total_x) / n
    print("R-K:", k, ",R-B:", b)


if __name__ == '__main__':
    x, y = gen_arr()
    least_squ(x, y)
