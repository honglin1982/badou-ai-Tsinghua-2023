# -*- coding: utf-8 -*-

import numpy as np
import cv2

# 手写双线性插值
# img：cv2.imread返回值
# out_dim：元组类型，目标图片宽高
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    (dst_h, dst_w) = out_dim
    # print ("src_h, src_w = ", src_h, src_w) # 512 512
    # print ("dst_h, dst_w = ", dst_h, dst_w) # 700, 700

    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(channel):
        for j in range(dst_h):
            for k in range(dst_w):

                # step1. 中心对称校准(公式有数学推导)，属于调整精度的优化
                src_x = (j + 0.5) * scale_x - 0.5
                src_y = (k + 0.5) * scale_y - 0.5

                # step2. 根据目标图，找到原图坐标，用于step3中带入双线性插值的公式来计算目标像素值
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1) # 避免边界溢出
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1) # 避免边界溢出

                # step3. 带入双线性插值公式求出虚拟像素点的值
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[k, j, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('./lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey(0)
