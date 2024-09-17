#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np


def bilinear_interp(src_img, dst_size):
    srch, srcw, channels = src_img.shape
    dsth, dstw = dst_size[1], dst_size[0]
    print("srch,srcw = ", srch, srcw)
    print("dsth,dstw = ", dsth, dstw)
    if srch == dsth and srcw == dstw:
        return src_img.copy()
    dst_img = np.zeros((dsth, dstw, channels), dtype=np.uint8)
    scale_x, scale_y = float(srcw) / dstw, float(srch) / dsth
    for i in range(channels):
        for dst_y in range(dsth):
            for dst_x in range(dstw):
                #  find the origin x and y coordinates of dst image x and y
                #  use geometric center symmetry
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))  # 向下取整
                src_x1 = min(src_x0 + 1, srcw - 1)  # 避免超出原数据范围
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, srch - 1)

                tmp0 = (src_x1 - src_x) * src_img[src_y0, src_x0, i] + (src_x - src_x0) * src_img[src_y0, src_x1, i]
                tmp1 = (src_x1 - src_x) * src_img[src_y1, src_x1, i] + (src_x - src_x0) * src_img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * tmp0 + (src_y - src_y0) * tmp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('/Users/aragaki/artificial/image/lenna.png')
    dst = bilinear_interp(img, (800, 800))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
