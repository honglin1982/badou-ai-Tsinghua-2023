# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:13:39 2023

@author: lhx
"""

import cv2
import numpy as np

#最邻近插值
def nearestInterpolation(img,targetH,targetW):
    h,w,c = img.shape
    newImg = np.zeros((targetH,targetW,c),np.uint8)
    
    for i in range(targetH):
        for j in range(targetW):
            #计算对应原图的位置 i:x=targetH:h,j:y=targetW:w
            #默认是向下取整，int(3.2)=3,int(3.7)=3，为了保证准确性（四舍五入），加0.5,int(3.2+0.5)=3,int(3.7+0.5)=4
            x=int(i/(targetH/h)+0.5)
            y=int(j/(targetW/w)+0.5)
            #print("计算结果："+str(x)+","+str(y))
            #判断是否超出边界，并修正
            x=x if x<h else h
            y=y if y<w else w
            #print("修正结果："+str(x)+","+str(y))
            
            newImg[i,j]=img[x,y]    
    return newImg
    
img=cv2.imread("lenna.png")#原图尺寸512*512
#放大后的图片
newImg1=nearestInterpolation(img, 800, 800)
#newImg2=nearestInterpolation(img, 1200, 1200)
#newImg3=nearestInterpolation(img, 1000, 600)
#newImg4=nearestInterpolation(img, 600, 800)

#opencv实现
#cv2.INTER_NEAREST	最近邻插值
#cv2.INTER_LINEAR	双线性插值（默认）
#cv2.INTER_AREA	使用像素区域关系进行重采样。
#cv2.INTER_CUBIC	4x4像素邻域的双3次插值
#cv2.INTER_LANCZOS4	8x8像素邻域的Lanczos插值
newImg5=cv2.resize(img,dsize=(800,800),interpolation=cv2.INTER_NEAREST)
newImg6=cv2.resize(img,dsize=(800,800),interpolation=cv2.INTER_LINEAR)
newImg7=cv2.resize(img,dsize=(800,800),interpolation=cv2.INTER_CUBIC)

#显示图片
cv2.imshow("orig",img)
cv2.imshow("800*800",newImg1)
#cv2.imshow("1200*1200",newImg2)
#cv2.imshow("1000*600",newImg3)
#cv2.imshow("600*800",newImg4)

cv2.imshow("cv2-nearest:800*800",newImg5)
cv2.imshow("cv2-linear:800*800",newImg6)
cv2.imshow("cv2-cubic:800*800",newImg7)

cv2.waitKey(0)
cv2.destroyAllWindows()