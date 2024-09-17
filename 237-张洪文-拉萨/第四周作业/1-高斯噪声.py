import cv2
import random
import numpy as np

"""
高斯噪声是一种常见的随机噪声类型，它基于正态分布（高斯分布）的随机数生成。其原理如下：
    正态分布：高斯噪声的生成基于正态分布（高斯分布），也称为钟形曲线。正态分布是一种连续概率分布，具有对称的钟形曲线形状。它由两个参数决定：
均值（mean）和标准差（standard deviation）。均值决定曲线的中心位置，标准差决定曲线的宽度。
    随机数生成：高斯噪声的生成过程涉及生成服从正态分布的随机数。常用的方法是使用随机数生成器函数，例如Python中的random.gauss()函数或
NumPy中的np.random.randn()函数。这些函数可以根据指定的均值和标准差生成符合正态分布的随机数。
    添加噪声：生成的随机数作为噪声信号，可以被添加到图像或其他信号中。通过将随机数加到原始信号的每个像素值上，可以在图像中引入高斯噪声。
高斯噪声的特点是在图像中引入随机的亮度变化，使得图像的像素值呈现连续的随机分布。由于正态分布的特性，大多数噪声值会集中在均值附近，而较大或较小的噪声值出现的概率较低。
在图像处理中，高斯噪声常用于模拟真实世界中的噪声情况，测试算法的鲁棒性和性能。同时，也可以通过滤波等技术来降低或去除高斯噪声对图像质量的影响。

实现流程：
a. 设定高斯函数的 means（平均值）、 sigma（标准方差）和决定进行高斯噪声化的图像像素个数百分比
b. 生成高斯随机数
d. 根据输入像素计算出输出像素
e. 重新将像素值放缩在[0 ~ 255]之间
f. 依次循环所有像素
g. 输出图像
"""

# 灰度图像
def gaussian_noise_gary(img, means, sigma, percentage):
    noise_img = img.copy()  # 创建原始图像的副本，而不是共享相同的内存空间
    print("原图像信息: ", img.shape, img.size)
    # 获取需要噪声化的像素数量
    noise_number = int(img.size * percentage)
    print("需要噪声化的像素数量: ", noise_number)

    # 从图像范围内随机选择指定数量的位置, replace=False 表示不允许重复选择同一个位置
    noise_positions = np.random.choice(img.size, size=noise_number, replace=False)

    # 遍历每一个需要噪声化的像素
    for pos in noise_positions:
        # 获取像素在图像中的索引,np.unravel_index用于将一维索引转换为多维索引,函数的返回值是一个元组，包含多维索引的值.
        rand_x, rand_y = np.unravel_index(indices=pos, shape=img.shape)
        # 生成高斯随机数并添加到像素值上
        noise = np.random.normal(loc=means, scale=sigma)
        noise_img[rand_x, rand_y] = np.clip(noise_img[rand_x, rand_y] + noise, a_min=0, a_max=255)

    return noise_img


# 彩色图像
def gaussian_noise_rgb(img, means, sigma, percentage):
    noise_img = img.copy()  # 创建原始图像的副本，而不是共享相同的内存空间
    print("原图像信息: ", img.shape, img.size)

    # 获取需要噪声化的像素数量
    noise_number = int(img.size * percentage)
    print("需要噪声化的像素数量: ", noise_number)

    # 从图像范围内随机选择指定数量的位置, replace=False 表示不允许重复选择同一个位置
    noise_positions = np.random.choice(img.size, size=noise_number, replace=False)

    # 遍历每一个需要噪声化的像素
    for pos in noise_positions:
        # 获取像素在图像中的索引,np.unravel_index用于将一维索引转换为多维索引,函数的返回值是一个元组，包含多维索引的值.
        rand_x, rand_y, channels = np.unravel_index(indices=pos, shape=img.shape)

        # 生成高斯随机数并添加到像素值上
        noise = np.random.normal(loc=means, scale=sigma)
        noise_img[rand_x, rand_y, channels] = np.clip(noise_img[rand_x, rand_y, channels] + noise, a_min=0, a_max=255)

    return noise_img


if __name__ == '__main__':
    # 灰度图像高斯噪声化
    # image = cv2.imread("./lenna.png", 0)  # 以灰度模式加载图像
    # img_gaussian = gaussian_noise_gary(image, means=2, sigma=15, percentage=1)  # 设定平均值、标准方差、噪声化比例

    # 彩色图像高斯噪声化
    image = cv2.imread("./lenna.png")
    img_gaussian = gaussian_noise_rgb(image, means=2, sigma=20, percentage=0.8)

    cv2.imshow('lenna_source', image)
    cv2.imshow('lenna_GaussianNoise', img_gaussian)
    # cv2.imwrite('lenna_GaussianNoise.png', img_gaussian)
    cv2.waitKey(0)
