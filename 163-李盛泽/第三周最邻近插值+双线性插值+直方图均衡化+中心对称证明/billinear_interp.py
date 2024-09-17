#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    用双线性插值求虚拟像素点的像素值
'''
import cv2
import numpy as np
from timeit import default_timer as timer


def nearest_interp(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_h / dst_h), float(src_w / dst_w)
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 1.根据目标图的(x,y)确定在原图中对应的一个虚拟像素点的坐标值
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 2.找到该虚拟像素点的邻近4点坐标，Q11,Q21,Q12,Q22
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)  # 在src_x0+1和src_w-1取更小值,限制边界在[src_w -1]
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 3.根据公式计算双线性插值
                temp_0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]  # (x1-x)fQ11 + (x-x0)fQ21
                temp_1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]  # (x1-x)fQ12 + (x-x0)fQ22
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp_0 + (src_y -src_y0) * temp_1)  # +0.5 ?

    return dst_img


if __name__ == '__main__':
    start = timer()
    img = cv2.imread('lenna.png', 1)
    dst = nearest_interp(img, (800, 800))
    cv2.imshow('billiner interp', dst)
    print("with billinear_interp: ", timer() - start)
    cv2.waitKey(0)