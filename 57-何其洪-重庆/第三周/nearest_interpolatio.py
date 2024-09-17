# -*- coding: utf-8 -*-

import numpy as np
import cv2


def nearestInterpolatio(img, zheight, zwidth):
    """
    最邻近插值缩放图片
    :param img: 图片
    :param zheight: 目标高度
    :param zwidth:  目标宽度
    :return: 缩放后的图片
    """
    height, width, channels = img.shape
    zoomInImg = np.zeros([zheight, zwidth, channels], dtype=np.uint8)
    sh = height/zheight
    sw = width/zwidth

    for row in range(zheight):
        for column in range(zwidth):
            x = int(row * sh + 0.5)
            y = int(column * sw + 0.5)
            zoomInImg[row, column] = img[x, y]
    return zoomInImg


if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png')
    zoomImg = nearestInterpolatio(img, 800, 800)
    cv2.imshow('img', zoomImg)
    cv2.waitKey(0)
