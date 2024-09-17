import numpy as np
import cv2
import matplotlib.pyplot as plt


def hist_eq(image):
    # 获取图像的直方图
    histogram, bin_edges = np.histogram(image, 256)
    # 获取累积分布函数
    cdf = histogram.cumsum()
    # 将累积分布函数转化为映射表
    cdf = 255 * cdf / cdf[-1]

    # 使用映射表执行直方图均衡化
    result = np.interp(image.flatten(), bin_edges[:-1], cdf)
    result = result.reshape(image.shape).astype("uint8")
    return result


if __name__ == "__main__":
    # 读取灰度图像
    img = cv2.imread("lenna.png", 0)
    # 显示直方图
    plt.figure()
    plt.hist(img.flatten(), 256)
    plt.show()

    # 执行直方图均衡化
    img1 = hist_eq(img)
    # 显示直方图
    # plt.figure()
    plt.hist(img1.ravel(), 256)
    plt.show()

    # 函数实现
    img2 = cv2.equalizeHist(img)

    # 展示图像
    cv2.imshow("Histogram Equalization", np.hstack([img, img1, img2]))
    cv2.waitKey()
"""
    # 读取彩色图像
    img = cv2.imread("lenna.png", 1)
    # 拆分三个通道
    b, g, r = cv2.split(img)

    # 执行直方图均衡化
    bH, gH, rH = hist_eq(b), hist_eq(g), hist_eq(r)
    img1 = cv2.merge((bH, gH, rH))

    # 函数实现
    bH, gH, rH = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
    img2 = cv2.merge((bH, gH, rH))

    cv2.imshow("Histogram Equalization", np.hstack([img, img1, img2]))
    cv2.waitKey()
"""