# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt


def grayHistogramFunc(image):
    """
    灰度图直方图均衡化
    :param image: 图片
    :return: 处理后的图片
    """
    # 1. 依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
    height, width = image.shape
    out = np.zeros([height, width], dtype=np.uint8)
    histogram = {}
    for row in range(height):
        for column in range(width):
            key = image[row, column]
            if key in histogram:
                histogram[key] += 1
            else:
                histogram[key] = 1
    # 展示直方图
    # plt.bar(range(len(histogram.keys())), [v[1] for v in sorted(histogram.items(), key=lambda d: d[0])])
    # plt.show()

    # 2. 计算灰度直方图的灰度分布频率
    keys = sorted(histogram.keys())
    pixel_count = height * width
    pi = []
    for key in keys:
        value = histogram[key]
        pi.append(value/pixel_count)

    # 3. 计算灰度直方图的累加直方图
    total = 0
    sum_pi_list = []
    for p in pi:
        total += p
        sum_pi_list.append((total * 256 - 1))

    # 4. 进行图像变换
    for row in range(height):
        for column in range(width):
            key = image[row, column]
            out[row, column] = max(round(sum_pi_list[keys.index(key)]), 0)
    return out


def colorHistogramFunc(image):
    """
    彩色图直方图均衡化
    :param image: 图片
    :return: 处理后的图片
    """
    # 1. 依次扫描原始图像的每个通道的每一个像素，计算出图像各通道的直方图
    height, width, channels = image.shape
    out = np.zeros([height, width, channels], dtype=np.uint8)
    for channel in range(channels):
        histogram = {}
        for row in range(height):
            for column in range(width):
                key = image[row, column, channel]
                if key in histogram:
                    histogram[key] += 1
                else:
                    histogram[key] = 1

        # 2. 计算该通道直方图的分布频率
        keys = sorted(histogram.keys())
        pixel_count = height * width
        pi = []
        for key in keys:
            value = histogram[key]
            pi.append(value/pixel_count)

        # 3. 计算直方图的累加直方图
        total = 0
        sum_pi_list = []
        for p in pi:
            total += p
            sum_pi_list.append((total * 256 - 1))

        # 4. 进行图像变换
        for row in range(height):
            for column in range(width):
                key = image[row, column, channel]
                out[row, column, channel] = max(round(sum_pi_list[keys.index(key)]), 0)
    return out


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    img = cv2.imread('../resources/images/lenna.png')
    # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 原图
    ax = plt.subplot(231)
    ax.set_title("灰度图")
    plt.imshow(gray_img, cmap="gray")
    ax = plt.subplot(232)
    ax.set_title("灰度直方图均衡化")
    plt.imshow(grayHistogramFunc(gray_img), cmap="gray")
    ax = plt.subplot(233)
    ax.set_title("API灰度直方图均衡化")
    plt.imshow(cv2.equalizeHist(gray_img), cmap="gray")
    ax = plt.subplot(234)
    ax.set_title("彩图")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax = plt.subplot(235)
    ax.set_title("彩图直方图均衡化")
    plt.imshow(cv2.cvtColor(colorHistogramFunc(img), cv2.COLOR_BGR2RGB))
    ax = plt.subplot(236)
    ax.set_title("API彩图直方图均衡化")
    (b, g, r) = cv2.split(img)
    bh = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    result = cv2.merge([bh, gh, rh])
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
