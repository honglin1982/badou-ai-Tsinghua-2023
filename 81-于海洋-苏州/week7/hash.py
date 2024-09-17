#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/6/14
@author: 81-于海洋
"""

import cv2

from config import Config


# 均值哈希算法
def avg_hash(img):
    """
    均值Hash算法
    步骤：
    1. 图片缩放到 8*8
    2. 灰度化
    3. 求平均值
    4. 比较 像素大于均值 记为1 小于均值设置为 0
    """
    # Step1. 图像缩放
    # cv2.INTER_NEAREST 最邻近插值， cv2.INTER_CUBIC 立方插值法
    # cv2.INTER_LINEAR 线性插值   cv2.INTER_LANCZOS4  Lanczos插值
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # Step2. 图像灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Step3. 求均值
    total = 0
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            total = total + gray[i, j]
    avg = total / 64
    hash_str = ''
    # Step4. 切换值大小
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值算法
def diff_hash(img):
    """
    插值Hash计算
    步骤：
    1. 图片缩放 8*9
    2. 转灰度图
    3. 每行 像素值大于后一个记作1 一行9个元素
    4. 生成hash
    """
    # Step1. 图片缩放
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # Step2. 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # Step3 + 4  生成Hash
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Hash值对比
def cmp_hash(hash1, hash2):
    count = 0
    if len(hash1) != len(hash2):
        print("hash len not eq")
        return -1

    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            count = count + 1
    return count


if __name__ == '__main__':
    """
    diff_hash 比 avg_hash 少了一次循环。
    相对时间复杂度上 lgO 和 lg2O的区别
    """
    img1 = cv2.imread(Config.LENNA)
    img2 = cv2.imread(Config.LENNA_BLUR)
    a_hash1 = avg_hash(img1)
    a_hash2 = avg_hash(img2)
    print(a_hash1)
    print(a_hash1)
    n = cmp_hash(a_hash1, a_hash1)
    print('A-Hash：', n)

    d_hash1 = diff_hash(img1)
    d_hash2 = diff_hash(img2)
    print(d_hash1)
    print(d_hash2)
    n = cmp_hash(d_hash1, d_hash2)
    print('D-Hash：', n)
