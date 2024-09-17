# -*- coding: utf-8 -*-
import cv2
import random


def deWeight(height, width, all):
    y = random.randint(0, height - 1)
    x = random.randint(0, width - 1)
    if y in all:
        if x in all.get(y):
            pass
        else:
            all.get(y).append(x)
    else:
        all[y] = [x]
    return y, x


def saltPepperNoise(img, ratio):
    """
    椒盐噪声
    :param img: 图片
    :param ratio: 噪声比例
    """
    height, width = img.shape[:2]
    piTotal = round(height * width * ratio)
    all = {}
    for i in range(piTotal):
        y, x = deWeight(height, width, all)
        img[y, x] = 0 if random.randint(0, 1) == 0 else 255


if __name__ == '__main__':
    original = cv2.imread('../resources/images/lenna.png')
    img = original.copy()
    saltPepperNoise(img, 0.01)
    cv2.imshow('original', original)
    cv2.imshow('img', img)
    cv2.waitKey(0)
