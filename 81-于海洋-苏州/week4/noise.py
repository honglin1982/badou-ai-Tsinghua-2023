# -*- coding: utf-8 -*-
"""
@date 2023/5/24
@author: y.haiyang@outlook.com
"""
import random

import cv2
import numpy as np
import skimage.util
from numpy import ndarray

from config import Config
from utils import ImgPreview


def mau_gs(src: ndarray, means, sigma, per):
    dest = np.copy(src)
    noise_num = int(src.shape[0] * src.shape[1] * per)
    for index in range(noise_num):
        x: int = random.randint(0, src.shape[0] - 1)
        y: int = random.randint(0, src.shape[1] - 1)
        value = dest[x, y] + random.gauss(means, sigma)
        if value < 0:
            value = 0
        elif value > 255:
            value = 255

        dest[x, y] = value

    preview.gray(dest, title="MU-Gaussian")


def mau_sp(src: ndarray, per):
    dest = np.copy(src)
    noise_num = int(src.shape[0] * src.shape[1] * per)
    for index in range(noise_num):
        x: int = random.randint(0, src.shape[0] - 1)
        y: int = random.randint(0, src.shape[1] - 1)
        if random.random() <= 0.5:
            value = 0
        else:
            value = 255

        dest[x, y] = value

    preview.gray(dest, title="MU-SP")


def noise_util():
    """
    通过工具实现图像加噪
    """
    dest = skimage.util.random_noise(img_grey, mode="speckle")
    preview.gray(dest, title="SK-Speckle")

    gs_noise = skimage.util.random_noise(img_grey, mode="gaussian")
    preview.gray(gs_noise, title="SK-Gaussian")

    sp_noise = skimage.util.random_noise(img_grey, mode="s&p")
    preview.gray(sp_noise, title="SK-S&P")

    dest = skimage.util.random_noise(img_grey, mode="localvar")
    preview.gray(dest, title="SK-Localvar")

    dest = skimage.util.random_noise(img_grey, mode="poisson")
    preview.gray(dest, title="SK-Poisson")


if __name__ == '__main__':
    img: ndarray = cv2.imread(Config.LENNA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    preview = ImgPreview(26, 10, 2, 4)
    preview.gray(img_grey, title="Original")
    noise_util()
    mau_gs(img_grey, 10, 20, 1)
    mau_sp(img_grey, 0.1)
    preview.show()
