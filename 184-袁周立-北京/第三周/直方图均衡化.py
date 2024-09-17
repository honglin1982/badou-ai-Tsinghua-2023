import cv2
import numpy as np

import matplotlib.pyplot as plt


'''
实现最近邻插值
实现双线性插值
证明几何中心对称系数
实现直方图均衡化
'''

def diy_equalization(img):
    src_h, src_w, channel = img.shape
    stat = np.zeros((channel, 256))   # 先记录0-255灰度值出现的次数，然后汇总算比例，最后记录对应均衡化后的灰度值
    dst_img = np.zeros((src_h, src_w, channel), dtype=np.uint8)

    for channel_index in range(channel):
        for i in range(src_h):
            for j in range(src_w):
                stat[channel_index][img[i, j, channel_index]] += 1
    plt.subplot(321)
    plt.plot([i for i in range(256)], stat[0], label="channel1")
    plt.subplot(323)
    plt.plot([i for i in range(256)], stat[1], label="channel2")
    plt.subplot(325)
    plt.plot([i for i in range(256)], stat[2], label="channel3")

    for channel_index in range(channel):
        for i in range(1, 256):
            stat[channel_index][i] += stat[channel_index][i - 1]
        for i in range(0, 256):
            stat[channel_index][i] = float(stat[channel_index][i]) / stat[channel_index][255]
            stat[channel_index][i] = max(0, int(stat[channel_index][i] * 256 - 1))
    for channel_index in range(channel):
        for i in range(src_h):
            for j in range(src_w):
                dst_img[i, j, channel_index] = stat[channel_index][img[i, j, channel_index]]

    stat = np.zeros((channel, 256))
    for channel_index in range(channel):
        for i in range(dst_img.shape[0]):
            for j in range(dst_img.shape[1]):
                stat[channel_index][dst_img[i, j, channel_index]] += 1
    plt.subplot(322)
    plt.plot([i for i in range(256)], stat[0], label="result_channel1")
    plt.subplot(324)
    plt.plot([i for i in range(256)], stat[1], label="result_channel2")
    plt.subplot(326)
    plt.plot([i for i in range(256)], stat[2], label="result_channel3")
    plt.show()

    return dst_img


img = cv2.imread("lenna.png")
diy_equalization_img = diy_equalization(img)
cv2.imshow("img", img)
cv2.imshow("diy_equalization_img", diy_equalization_img)
cv2.waitKey()

