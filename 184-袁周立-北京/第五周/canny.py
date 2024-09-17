import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


'''
实现CANNY 实现透视变换
'''
def Canny_DIY(img, gauss_kernal_size=5, sigma=1):
    h, w, channel = img.shape
    plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),  plt.title("origin")

    # 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.subplot(232), plt.imshow(img_gray, cmap="gray"), plt.title("gray")

    # 高斯平滑
    gauss_kernal = np.zeros((gauss_kernal_size, gauss_kernal_size))
    center_index = (gauss_kernal_size - 1) // 2
    param1 = 1 / (2 * math.pi * sigma ** 2)
    param2 = -1 / (2 * sigma ** 2)
    for i in range(gauss_kernal_size):
        for j in range(gauss_kernal_size):
            tmp = (i - center_index) ** 2 + (j - center_index) ** 2
            gauss_kernal[i, j] = param1 * math.exp(param2 * tmp)
    gauss_kernal = gauss_kernal / np.sum(gauss_kernal)
    pad_size = center_index
    img_pad = np.pad(img_gray, ((pad_size, pad_size), (pad_size, pad_size)), "constant")
    img_gaussfilter = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            img_pad_kernel = img_pad[i:i+gauss_kernal_size, j:j+gauss_kernal_size]
            img_gaussfilter[i, j] = np.sum(np.multiply(img_pad_kernel, gauss_kernal))

    plt.subplot(233), plt.imshow(img_gaussfilter.astype(np.uint8), cmap="gray"), plt.title("img_gaussfilter")

    # sobel算子
    # x方向从左往右，y方向从下往上，这个方向与之后非极大值抑制算梯度方向相对应
    # 即y/x在0到1之间时，说明梯度方向的直线与水平方向之间的夹角在0到45°，从而确定该直线与8领域相交的点的位置
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    img_tidu_x = np.zeros((h, w))
    img_tidu_y = np.zeros((h, w))
    img_tidu = np.zeros((h, w))
    img_tidu_angle = np.zeros((h, w))
    for i in range(1, h-1):
        for j in range(1, w-1):
            tmp = img_gaussfilter[i-1:i+2, j-1:j+2]
            img_tidu_x[i, j] = np.sum(np.multiply(tmp, sobel_x))
            img_tidu_y[i, j] = np.sum(np.multiply(tmp, sobel_y))
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
            if img_tidu_x[i, j] != 0:
                img_tidu_angle[i, j] = img_tidu_y[i, j] / img_tidu_x[i, j]
            else:
                img_tidu_angle[i, j] = 999999

    plt.subplot(234), plt.imshow(img_tidu.astype(np.uint8), cmap="gray"), plt.title("img_tidu")

    # 非极大值抑制
    img_yizhi = np.zeros((h, w))
    for i in range(1, h-1):
        for j in range(1, w-1):
            tmp = img_tidu[i-1:i+2, j-1:j+2]
            angle = img_tidu_angle[i, j]
            tidu1, tidu2 = 0, 0
            if angle > 1:
                tidu1 = tmp[0, 1] * (1 - 1 / angle) + tmp[0, 2] / angle
                tidu2 = tmp[2, 0] * (1 / angle) + tmp[2, 1] * (1 - 1 / angle)
            elif angle > 0:
                tidu1 = tmp[0, 2] * angle + tmp[1, 2] * (1 - angle)
                tidu2 = tmp[1, 0] * (1 - angle) + tmp[2, 0] * angle
            elif angle > -1:
                tidu1 = tmp[1, 2] * (1 + angle) + tmp[2, 2] * (-angle)
                tidu2 = tmp[0, 0] * (-angle) + tmp[1, 0] * (1 + angle)
            else:
                tidu1 = tmp[0, 0] * (-1 / angle) + tmp[0, 1] * (1 + 1 / angle)
                tidu2 = tmp[2, 1] * (1 + 1 / angle) + tmp[2, 2] * (-1 / angle)
            if img_tidu[i, j] > tidu1 and img_tidu[i, j] > tidu2:
                img_yizhi[i, j] = img_tidu[i, j]

    plt.subplot(235), plt.imshow(img_yizhi.astype(np.uint8), cmap="gray"), plt.title("img_yizhi")

    # 双阈值检测
    threshold_low = np.mean(img_tidu) / 2
    threshold_high = threshold_low * 3
    threshold_high_list = []
    img_final = np.copy(img_yizhi)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img_final[i, j] >= threshold_high:
                img_final[i, j] = 255
                threshold_high_list.append([i, j])
    while len(threshold_high_list) != 0:
        i, j = threshold_high_list.pop()
        index_list = [[i - 1, j - 1], [i - 1, j], [i - 1, j + 1],
                      [i, j - 1], [i, j + 1],
                      [i + 1, j - 1], [i + 1, j], [i + 1, j + 1]]
        for index in index_list:
            if threshold_low < img_final[index[0], index[1]] < threshold_high:
                img_final[index[0], index[1]] = 255
                threshold_high_list.append([index[0], index[1]])
    img_final[img_final != 255] = 0

    plt.subplot(236), plt.imshow(img_final.astype(np.uint8), cmap="gray"), plt.title("img_final")
    plt.show()
    return img_final


img_read = cv2.imread("lenna.png")
img_canny = Canny_DIY(img_read)
cv2.imshow("img_canny", img_canny)
cv2.waitKey()