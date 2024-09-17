# -*- coding: utf-8 -*-
import cv2
import random


def gaussianNoise(gray_img, ratio, mu, sigma):
    """
    图片高斯噪声
    :param gray_img: 灰度图
    :param ratio: 覆盖比例
    """
    height, width = gray_img.shape
    piTotal = round(height * width * ratio)
    for i in range(piTotal):
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        result = gray_img[y, x] + random.gauss(mu, sigma)
        if result < 0:
            result = 0
        if result > 255:
            result = 255
        gray_img[y, x] = result


if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianNoise(gray_img, 1, 3, 4)
    cv2.imshow('original', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cv2.imshow('img', gray_img)
    cv2.waitKey(0)
