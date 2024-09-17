import cv2
import random

import numpy as np


def salt_and_pepper_noise(src_img, percetage):
    NoiseImg = src_img
    NoiseNum = int(percetage * src_img.shape[0] * src_img.shape[1])  # 加噪声的百分比
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        randX = random.randint(0, src_img.shape[0] - 1)
        randY = random.randint(0, src_img.shape[1] - 1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


# 另一种方式实现椒盐噪声
def salt_and_pepper_noise_1(image, prob):
    """
        噪声函数
       :param image: 原图像，是灰度图
       :param prob: 控制椒盐噪声的数量，这里是0-1的一个概率值
       :return: 处理后的图像
    """
    h, w = image.shape[:2]
    noise = np.zeros((h, w), dtype=np.uint8)
    # 将noise随机填充0-255的值
    cv2.randu(noise, 0, 255)
    # 将image传给image_copy
    image_copy = image.copy()
    # prob*255就是我们选的那个阈值
    image_copy[np.where(noise < prob * 255)] = 0
    image_copy[np.where(noise > (1 - prob) * 255)] = 255
    return image_copy


img = cv2.imread('../data/lenna.png')
result = salt_and_pepper_noise(img, 0.2)

img = cv2.imread('../data/lenna.png')
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度图
cv2.imshow('source', img)
cv2.imshow('lenna_PepperandSalt', result)
cv2.imshow('img', np.hstack([img, result]))  # 放在一起对比
cv2.waitKey(0)
