# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2023/4/23 22:22
# @File    : double_linear_interpolation.py
# @Software: PyCharm
import time

import cv2
import numpy as np
import cv2 as cv
#双线性插值，需要进行3个通道的遍历。
def double_linear_interpolation_3for(img , rs_h , rs_w ,is_gray=False):
    assert img is not None ,"图像不存在哦"
    if (is_gray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img_gray[:, :, np.newaxis]
    h , w, c = img.shape
    assert rs_h>h and rs_w>w , "插值后图像尺寸应当大于原始图像哦"
    if(h==rs_h and w==rs_w):
        return img.copy()
    interpolate_img = np.zeros((rs_h,rs_w,c),np.uint8)
    #这里找到转变尺寸，不用整除
    trans_size_h = rs_h / float(h)
    trans_size_w = rs_w / float(w)
    for k in range(c):
        for i in range(rs_h):
            for j in range(rs_w):
                #进行中心对齐 中心对齐减0.5
                inter_x = (i + 0.5) / trans_size_h - 0.5
                inter_y = (j + 0.5) / trans_size_w - 0.5
                #确定插值周围的四个点，左上右下，左上下取整，右下+1
                x0 = int(np.floor(inter_x))
                x1 = min(x0 + 1 , h-1)
                y0 = int(np.floor(inter_y))
                y1 = min(y0 + 1 , w-1)
                #两次单线性完成双线性
                t0 = (x1 - inter_x) * img[x0, y0, k] + (inter_x - x0) * img[x1, y0, k]
                t1 = (x1 - inter_x) * img[x0, y1, k] + (inter_x - x0) * img[x1, y1, k]
                interpolate_img[i, j, k] = int((y1 - inter_y) * t0 + (inter_y - y0) * t1)
    return interpolate_img

def double_linear_interpolation_2for(img , rs_h , rs_w ,is_gray=False):
    assert img is not None ,"图像不存在哦"
    if (is_gray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img_gray[:, :, np.newaxis]
    h , w, c = img.shape
    assert rs_h>h and rs_w>w , "插值后图像尺寸应当大于原始图像哦"
    if(h==rs_h and w==rs_w):
        return img.copy()
    interpolate_img = np.zeros((rs_h,rs_w,c),np.uint8)
    #这里找到转变尺寸，不用整除
    trans_size_h = rs_h / float(h)
    trans_size_w = rs_w / float(w)
    #全是整数操作，就不用遍历通道了
    for i in range(rs_h):
        for j in range(rs_w):
            #进行中心对齐 中心对齐减0.5
            inter_x = (i + 0.5) / trans_size_h - 0.5
            inter_y = (j + 0.5) / trans_size_w - 0.5
            #确定插值周围的四个点，左上右下，左上下取整，右下+1
            x0 = int(np.floor(inter_x))
            x1 = min(x0 + 1 , h-1)
            y0 = int(np.floor(inter_y))
            y1 = min(y0 + 1 , w-1)
            #两次单线性完成双线性
            t0 = (x1 - inter_x) * img[x0, y0] + (inter_x - x0) * img[x1, y0]
            t1 = (x1 - inter_x) * img[x0, y1] + (inter_x - x0) * img[x1, y1]
            interpolate_img[i, j] = (y1 - inter_y) * t0 + (inter_y - y0) * t1
    return interpolate_img

if __name__ == '__main__':
    img = cv.imread("lane.jpg")
    start = time.perf_counter()
    #使用3次循环得到插值结果，时间16s
    #使用2次循环得到插值结果，时间7s
    interpolate_img = double_linear_interpolation_3for(img,800,800,is_gray=False)
    end = time.perf_counter()
    print(img.shape)
    print(interpolate_img.shape)
    print(f'函数执行时间：{end - start:.6f}s')
    cv2.imshow("img",img)
    cv2.imshow("interpolate_img",interpolate_img)
    cv2.waitKey(0)

