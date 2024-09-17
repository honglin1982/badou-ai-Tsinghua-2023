# -*- coding: utf-8 -*-
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def gaussianKernel(kernel_size, sigma):
    """
    获取高斯核
    :param kernel_size: 高斯核大小
    :return: 高斯核
    """
    gaussian_kernel = np.zeros([kernel_size, kernel_size])
    # 用于坐标转换
    pad = int(kernel_size / 2)
    for y in range(kernel_size):
        for x in range(kernel_size):
            # 计算出坐标
            _y = y - pad
            _x = x - pad
            gaussian_kernel[y, x] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(_y ** 2 + _x ** 2) / (2 * sigma ** 2))
    # 归一化
    return gaussian_kernel / gaussian_kernel.sum()


def gaussianFilter(img, kernel_size, sigma):
    """
    对图像进行高斯平滑
    :param img: 图片
    :return: 处理后的图片
    """
    pad = int(kernel_size / 2)
    gaussian_kernel = gaussianKernel(kernel_size, sigma)
    print('高斯核：\n', gaussian_kernel)
    height, width = img.shape
    img_gaussian = np.zeros(img.shape)
    # 对图像进行padding操作
    # np.pad(array, pad_width, mode)
    # array表示需要填充的数组，
    # np.pad(array, ((2,1), (1, 2)), 'constant')
    # pad_width表示每个轴需要填充的数目，(2,1) 表示在横轴上方填充两行，下方填充1行；(1, 2) 表示在纵轴左方填充1列，右方填充2列
    #          简写：(2,3)表示上下各填充2行，左右各填充2列
    #               2 表示上下左右各填充2行/列
    # mode: constant表示连续填充相同的值，constant_values=(x, y)时前面用x填充，后面用y填充，缺省填充0
    img_pad = np.pad(img, pad, 'constant')
    print("padding: \n", img)
    for y in range(height):
        for x in range(width):
            # 切片的使用，[行进行切片,列进行切片] 即[start:stop:step,start:stop:step]
            box = img_pad[y:y + kernel_size, x:x + kernel_size]
            img_gaussian[y, x] = np.sum(box * gaussian_kernel)
    return img_gaussian


if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png')
    # 1. 灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax = plt.subplot(221)
    ax.set_title("灰度图")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    # 2. 高斯滤波
    img_gaussian = gaussianFilter(img, 5, 0.5)
    ax = plt.subplot(222)
    ax.set_title("高斯滤波")
    plt.imshow(img_gaussian, cmap='gray')
    plt.axis('off')
    # 3. 检测图像中的水平、垂直和对角边缘，使用Sobel算子
    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    kernel_size = 3
    pad = int(kernel_size / 2)
    height, width = img_gaussian.shape
    img_sobel_x = np.zeros(img_gaussian.shape)
    img_sobel_y = img_sobel_x.copy()
    img_gradient = img_sobel_x.copy()
    img_pad = np.pad(img_gaussian, pad)
    for y in range(height):
        for x in range(width):
            box = img_pad[y:y + kernel_size, x:x + kernel_size]
            """
            当使用sobel_kernel_x时，会发现中间列全部为0，不参与计算，
            实际上是右边一列减去左边一列，最终计算的是【水平方向上的梯度】
            """
            img_sobel_x[y, x] = np.sum(box * sobel_kernel_x)
            # 同理sobel_kernel_y计算的是垂直方向上的梯度
            img_sobel_y[y, x] = np.sum(box * sobel_kernel_y)
            # 由勾股定理可得梯度值
            img_gradient[y, x] = np.sqrt(img_sobel_x[y, x] ** 2 + img_sobel_y[y, x] ** 2)
    # 三角函数tanθ = y / x 用于计算梯度方向θ
    tan_theta = img_sobel_y / img_sobel_x
    # 4. 对梯度幅值进行非极大值抑制
    img_suppress = np.zeros(img_gradient.shape)
    # 因为需要取周围的八个点，所以从下标1开始到最后一个下标-1结束
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            d1 = 0
            d2 = 0
            # 取出3x3
            tmp = img_gradient[y - 1:y + 2, x - 1:x + 2]
            """
            canny主要对四个方向的梯度进行抑制（0°、45°、90°、135°）
            现实情况下图像中的梯度方向不一定是沿着这四个方向的，
            因此使用插值求出梯度相交的两个点进行比较判断是否抑制
            tan45°=1,tan135=-1
            
            A1 A2 A3
            A4 A5 A6
            A7 A8 A9
            """
            tt = math.fabs(tan_theta[y, x])
            if tan_theta[y, x] <= -1:
                """
                此时A2和A8在垂直方向
                插值表示为: d1 = (A1-A2) / tanθ + A2
                          d2 = (A9-A8) / tanθ + A8
                """
                d1 = (tmp[0, 0] - tmp[0, 1]) / tt + tmp[0, 1]
                d2 = (tmp[2, 2] - tmp[2, 1]) / tt + tmp[2, 1]
            elif tan_theta[y, x] >= 1:
                """
                此时A2和A8在垂直方向
                插值表示为: d1 = (A3-A2) / tanθ + A2
                          d2 = (A7-A8) / tanθ + A8
                """
                d1 = (tmp[0, 2] - tmp[0, 1]) / tt + tmp[0, 1]
                d2 = (tmp[2, 0] - tmp[2, 1]) / tt + tmp[2, 1]
            elif tan_theta[y, x] > 0:
                """
                此时A4和A6在水平方向
                插值表示为: d1 = (A7-A4) * tanθ + A4
                          d2 = (A3-A6) * tanθ + A6
                """
                d1 = (tmp[2, 0] - tmp[1, 0]) * tt + tmp[1, 0]
                d2 = (tmp[0, 2] - tmp[1, 2]) * tt + tmp[1, 2]
            elif tan_theta[y, x] < 0:
                """
                此时A4和A6在水平方向
                插值表示为: d1 = (A1-A4) * tanθ + A4
                          d2 = (A9-A6) * tanθ + A6
                """
                d1 = (tmp[0, 0] - tmp[1, 0]) * tt + tmp[1, 0]
                d2 = (tmp[2, 2] - tmp[1, 2]) * tt + tmp[1, 2]

            # 判断是否不需要被抑制
            if tmp[1, 1] > d1 and tmp[1, 1] > d2:
                img_suppress[y, x] = tmp[1, 1]
    ax = plt.subplot(223)
    ax.set_title("非极大值抑制")
    plt.imshow(img_suppress.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # 5. 双阈值算法检测，低于低阈值将被抑制，高于高阈值将被增强，中间值将判断是否连接
    # 设定高低阈值
    lower_threshold = img_gradient.mean() / 2
    high_threshold = lower_threshold * 3
    # 用于保存高阈值的坐标，方便连接边缘
    stack = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # 双阈值检测
            if img_suppress[y, x] >= high_threshold:
                img_suppress[y, x] = 255
                stack.append([y, x])
            elif img_suppress[y, x] <= lower_threshold:
                img_suppress[y, x] = 0

    while len(stack) > 0:
        y, x = stack.pop()
        # 遍历极大值周围的八个点
        for y1 in range(3):
            for x1 in range(3):
                # 计算出坐标
                _y = y1 - 1
                _x = x1 - 1
                if _y == 0 and _x == 0:
                    continue
                _y = y + _y
                _x = x + _x
                if high_threshold > img_suppress[_y, _x] > lower_threshold:
                    img_suppress[_y, _x] = 255
                    stack.append([_y, _x])
    for y in range(height):
        for x in range(width):
            if img_suppress[y, x] != 0 and img_suppress[y, x] != 255:
                img_suppress[y, x] = 0
    ax = plt.subplot(224)
    ax.set_title("结果")
    plt.imshow(img_suppress, cmap='gray')
    plt.axis('off')
    plt.show()
