import cv2
import numpy as np
from matplotlib import pyplot as plt


# 手工实现灰度化
def graying_1():
    # 1、读取图像，返回一个表示图像的多维数组（即numpy数组），数组的维度对应于图像的高度、宽度和通道数
    img = cv2.imread("data/lenna.png")
    # print(type(img.shape), img.shape)

    # 2、切片操作 [:2] 获取元组的前两个元素（高度和宽度）
    h, w = img.shape[:2]
    # print(h, w)

    # 3、创建一个与当前图像大小相同的全零NumPy数组,这个数组将用于存储灰度图像的像素值
    img_gray = np.zeros(shape=(h, w), dtype=img.dtype)
    # print(img.dtype)  # uint8 无符号8位整数

    # 4、遍历图像的每个像素点（即每个数组值）, 进行重新计算赋值
    for i in range(h):
        for j in range(w):
            # 注意：opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB。
            # 这里的 img 是一个三维数组, img[i, j] 为一个一维数组，有三个元素，对应 B G R
            m = img[i, j]
            # 将原先的BGR数据，通过浮点算法灰度化：Gray = R0.3 + G0.59 + B0.11
            img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

    # 5、保存图像并展示
    cv2.imwrite("data/lenna_gray.png", img_gray)

    # 6、展示图像
    plt.subplot(121)
    # 注意，plt 读取的图像是进行了归一化操作了的，cv读取的图像为原始范围的0~255
    img = plt.imread("data/lenna.png")
    plt.imshow(img)
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Gray Image")

    plt.show()

# 接口实现灰度化
def graying_2():
    # 读取图像
    img = cv2.imread("data/lenna.png")
    # 原始图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 保存图片
    cv2.imwrite("data/lenna_gray2.png", img_gray)

    plt.subplot(121)
    plt.imshow(img_rgb)
    plt.title("Original Image")

    plt.subplot(122)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Gray Image")

    plt.show()


if __name__ == '__main__':
    graying_1()
    graying_2()
