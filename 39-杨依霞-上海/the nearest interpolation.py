# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:16:20 2023

@author: YYX
"""

#The nearest interpolation
import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
#定义高、宽和通道
    emptyImage=np.zeros((800,800,channels),np.uint8)
#建立一个空的图像，放大至800*800（原512*512），通道保持一致
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
#横向和纵向遍历所有像素点
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
#把原图的像素点坐标转化成定点（向上取整）
            emptyImage[i,j]=img[x,y]
#找出原图对应的像素点坐标
    return emptyImage
        
img=cv2.imread("G:/test/lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)