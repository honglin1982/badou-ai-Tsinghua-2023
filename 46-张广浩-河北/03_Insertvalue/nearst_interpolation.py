# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2023/4/23 21:20
# @File    : nearst_interpolation.py
# @Software: PyCharm
import math

import cv2
import numpy as np
import cv2 as cv

#这里最近临插值可以直接处理灰度图和彩色图，因为3个通道直接能够赋予插值后的目标图。
#所以不用遍历一遍通道维度
def nearst_interpolation(img , rs_h , rs_w, is_gray=False):
    #图像需要存在
    assert img is not None ,"图像不存在哦"
    #灰度化处理以及保留维度
    if(is_gray):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img_gray[:,:,np.newaxis]
    h , w, c= img.shape
    #插值图像大于原始图像
    assert rs_h>h and rs_w>w , "插值后图像尺寸应当大于原始图像哦"
    #初始化
    interpolate_img = np.zeros((rs_h,rs_w,1),np.uint8)
    #这里找到转变尺寸，不用整除
    trans_size_h = rs_h / h
    trans_size_w = rs_w /w
    #这里遍历长宽维度即可
    for i in range(rs_h):
        for j in range(rs_w):
            u = int(i/trans_size_h + 0.5)
            #控制对应的像素不超过原始图像像素
            u = min(u, h - 1)
            v = int(j/trans_size_w + 0.5)
            v = min(v, w - 1)
            #单通道直接赋值，多通道3个通道也直接复值
            interpolate_img[i,j] = img[u,v]
    return interpolate_img

if __name__ == '__main__':
    img = cv.imread("lane.jpg")
    interpolate_img = nearst_interpolation(img,800,800,is_gray=True)
    print(img.shape)
    print(interpolate_img.shape)
    cv2.imshow("img",img)
    cv2.imshow("interpolate_img",interpolate_img)
    cv2.waitKey(0)

