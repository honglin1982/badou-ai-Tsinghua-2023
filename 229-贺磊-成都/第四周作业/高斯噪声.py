# 随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
import random


def GaussianNoise(src_img, means, sigma, percentage):
    """

    :param src_img:  原图
    :param means: 高斯均值
    :param sigma: 高斯σ
    :param percentage: 百分比
    :return: 加噪之后的图
    """
    NoiseImg = src_img
    NoiseNum = int(percentage * src_img.shape[0] * src_img.shape[1])
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX = random.randint(0, src_img.shape[0] - 1)
        randY = random.randint(0, src_img.shape[1] - 1)
        # 此处在原有像素灰度值上加上随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if NoiseImg[randX, randY].all() < 0:
            """.all()查阅资料得知在numpy中矩阵的比较是依次进行的（即每个位置的数对应比较），
            所以两个矩阵比较时要想判断是否完全相等应该在后面加上.all()
            若要判断两个矩阵中是否存在一个值相等用.any()
            """
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY].all() > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg

# 高斯噪声
def gaussian_noise(image, mean=0, var=0.1):
    """
    给输入的图像添加高斯噪声
    :param image: 输入图像，0-255的灰度图
    :param mean: 高斯噪声的均值，默认为0
    :param var: 高斯噪声的标准差，默认为0.1
    :return: 添加高斯噪声后的图像
    """
    # 先把图像转化为0-1，并将类型转化为float32，这样有利于保存数据，
    image = np.asarray(image / 255, dtype=np.float32)
    # 为了后面加方便，noise也要转化为float32
    noise = np.random.normal(mean, var, image.shape).astype(np.float32)
    noisy_image = image + noise
    # 将noise_image限制在0-255，因为+运算可能有部分会超过255，再转化为整型
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    return noisy_image


img = cv2.imread('../data/lenna.png')
result = GaussianNoise(img, 2, 4, 0.8)
res = gaussian_noise(img)
# img = cv2.imread('lenna.png')
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('lenna_GaussianNoise.png',result)
# cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', np.hstack([img,result,res]))

cv2.waitKey(0)



