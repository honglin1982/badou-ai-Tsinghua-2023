import cv2
import numpy as np
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.inf)  # 设置打印数组时不忽略任何元素


# 二值化
def binary():
    # cv2.IMREAD_GRAYSCALE设置以灰度模式读入，返回一个二维numpy数组
    img_gray = cv2.imread("data/lenna_gray.png", cv2.IMREAD_GRAYSCALE)

    # 第一种方法
    rows, cols = img_gray.shape  # 获取行、列数
    img_binary = np.zeros(img_gray.shape, dtype=img_gray.dtype)
    for i in range(rows):   # 循环元素，通过判断设置二值：255 0
        for j in range(cols):
            if int(img_gray[i, j]) >= 128:
                img_binary[i, j] = 255
            else:
                img_binary[i, j] = 0

    # 第二种方法：使用numpy函数进行判断，其用于根据给定的条件返回符合条件的元素的索引或值。
    img_binary = np.where(img_gray >= 128, 255, 0)

    # 第三种方法：通过OpenCV提供的接口函数 cv2.threshold
    # 函数返回两个值，其中第一个值是使用的阈值（在本例中为128），第二个值是二值化后的图像
    _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    # 显示图像
    plt.subplot(121)
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(122)
    plt.imshow(img_binary, cmap='gray')
    plt.show()

    # 保存图像
    cv2.imwrite("data/lenna_binary.png", img_binary)


if __name__ == '__main__':
    binary()
