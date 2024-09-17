# -*- coding: utf-8 -*-
# @Author  : ZGH
# @Time    : 2023/4/23 23:12
# @File    : Histogram_equalization.py
# @Software: PyCharm

#获得灰度直方图
import cv2
import numpy as np
from matplotlib import pyplot as plt


def hist(img):
    #二维像素矩阵拉伸为一维
    img_ravel = img.ravel()
    #初始化数组（0-255）
    hist = [0 for i in range(256)]
    #遍历像素点，进行相同像素值累加
    for p in img_ravel:
        hist[int(p)] +=1
    #进行归一化
    # for i in range(256):
    #     hist[i] /=img_ravel.shape[0]
    return hist

def sum_hist(hist):
    sh = [0 for i in range(256)]
    sh[0] = hist[0]
    for i in range(1,256):
        sum = sh[i-1] + hist[i]
        sh[i] = sum
    return sh

def histogram_equalization(img_gray):
    h, w = img_gray.shape
    hi = hist(img_gray)
    sh = sum_hist(hi)
    he_img = np.zeros_like(img_gray)
    for i in range(h):
        for j in range(w):
            p = int((sh[img_gray[i,j]] / (h*w)) * 256 - 1)
            he_img[i,j] = p

    return he_img

def histogram_equalization_color(img):
    h, w, c = img.shape
    his = []
    shs = []
    for i in range(3):
        hi = hist(img[:,:,i])
        sh = sum_hist(hi)
        his.append(hi)
        shs.append(sh)
    hw = h * w
    he_img = np.zeros_like(img)
    for k in range(c):
        for i in range(h):
            for j in range(w):
                s = shs[k][img[i,j,k]]
                p = int(( s / hw) * 256 - 1)
                he_img[i,j,k] = p
    return he_img

if __name__ == '__main__':
    img_gray = cv2.imread("lane.jpg",0)
    img = cv2.imread("photo.jpg")

    # img = cv2.cvtColor(img,code=cv2.COLOR_BGR2RGB)
    #cv2.imshow("img_gray",img_gray)
    cv2.imshow("img", img)
    he = histogram_equalization_color(img)
    he_img_self = histogram_equalization(img_gray)
    he_img = cv2.equalizeHist(img_gray)
    cv2.imshow("he", he)
    #cv2.imshow("he_img_self",he_img_self)
    #cv2.imshow("he_img", he_img)
    cv2.imwrite("he.jpg",he)
    cv2.waitKey(0)
    # h = hist(img_gray)
    # sh = sum_hist(h)
    # plt.plot(h)
    # plt.plot(sh)
    # plt.show()
