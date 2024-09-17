# -*- coding: gbk -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 单通道直方图均衡化: equalizeHist
def histogram_equalization_1(img):
    # 将图像转为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.equalizeHist 该函数接受灰度图像作为输入，并返回进行直方图均衡化后的图像。
    equalized = cv2.equalizeHist(gray_img)

    # cv2.calcHist()函数接收一幅图像作为输入，并返回计算得到的直方图。
    img_hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])

    # plt.figure()  # 创建一个新的图形窗口，用于显示直方图
    plt.subplot(121)
    plt.plot(img_hist)

    plt.subplot(122)
    # 使用plt.hist()函数绘制直方图, equalized.ravel()将多维数组展平为一维数组，以便作为直方图的输入。256表示直方图的bin数，即将灰度值范围划分为256个区间。
    plt.hist(equalized.ravel(), 256)
    plt.show()  # 显示绘制的直方图图形窗口

    # np.hstack([gray_img, equalized])将原始灰度图像gray和经直方图均衡化后的灰度图像dst水平堆叠在一起。
    cv2.imshow("Histogram Equalization", np.hstack([gray_img, equalized]))
    cv2.waitKey(0)

# 多通道直方图均衡化
def histogram_equalization_2(img):
    # 将图像转换为RGB颜色空间
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 分割颜色通道
    r, g, b = cv2.split(rgb_img)

    # 对每个颜色通道进行直方图均衡化
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # 合并均衡化后的颜色通道
    equalized_img = cv2.merge([r_eq, g_eq, b_eq])

    # 显示原始图像和均衡化后的图像
    plt.subplot(121)
    plt.imshow(rgb_img)
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(equalized_img)
    plt.title("Equalized Image")

    plt.show()


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # histogram_equalization_1(img)
    histogram_equalization_2(img)
