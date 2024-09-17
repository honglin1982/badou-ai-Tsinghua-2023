# -*- coding: utf-8 -*-
import numpy as np
import cv2


def bilinearInterpolation(img, zheight, zwidth):
    """
    双线性插值缩放图片
    :param img: 图片
    :param zheight: 目标高度
    :param zwidth:  目标宽度
    :return: 缩放后的图片
    """
    height, width, channels = img.shape
    zoomImg = np.zeros([zheight, zwidth, channels], dtype=np.uint8)
    sw = width / zwidth
    sh = height / zheight
    for channel in range(channels):
        for row in range(zheight):
            for column in range(zwidth):
                # 几何中心重合
                src_x = (column + 0.5) * sw - 0.5
                src_y = (row + 0.5) * sh - 0.5

                src_x0 = int(src_x)
                src_y0 = int(src_y)
                src_x1 = min(src_x0 + 1, width - 1)
                src_y1 = min(src_y0 + 1, height - 1)
                # 计算插值
                r1 = (src_x1 - src_x) * img[src_y0, src_x0, channel] + (src_x - src_x0) * img[src_y0, src_x1, channel]
                r2 = (src_x1 - src_x) * img[src_y1, src_x0, channel] + (src_x - src_x0) * img[src_y1, src_x1, channel]
                zoomImg[row, column, channel] = (src_y1 - src_y) * r1 + (src_y - src_y0) * r2
    return zoomImg


if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png')
    zoomImg = bilinearInterpolation(img, 800, 800)
    cv2.imshow('img', zoomImg)
    cv2.waitKey(0)
