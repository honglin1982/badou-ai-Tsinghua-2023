# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 11:30:42 2023

@author: lhx
"""

import cv2
import numpy as np

#双线性插值
def linearInterpolation(img,dstH,dstW):
    h,w,c = img.shape
    if h==dstH and w==dstW:
        return img
    newImg = np.zeros((dstH,dstW,c),np.uint8)
    
    for i in range(c):#通道
        for j in range(dstH):#行/高
            for k in range(dstW):#列/宽
                #使几何中心相等，求出目标点对应的原图的点 src:源；dst:目的
                #srcX+0.5=(dstX+0.5)*(srcW/dstW) 或写成 (srcX+0.5)/(dstX+0.5)=srcW/dstW
                srcX = (k+0.5)*(float(w)/dstW)-0.5#转浮点型，提升精度
                srcY = (j+0.5)*(float(w)/dstW)-0.5#转浮点型，提升精度
                #求出2个点用于计算
                srcX1 = int(srcX)#向下取整作为第一个点
                srcX2 = min(srcX1+1, w-1)#加一个像素作为第二个点，防止越界
                srcY1 = int(srcY)#向下取整作为第一个点
                srcY2 = min(srcY1+1, h-1)#加一个像素作为第二个点，防止越界
                
                #公式：(y2-y)*((x2-x)*img[y1,x1,c]+(x-x1)*img[y1,x2,c])+(y-y1)*((x2-x)*img[y2,x1,c]+(x-x1)*img[y2,x2,c])
                # 其实是4个点的比重相加
                v1 = (srcY2-srcY)*(srcX2-srcX)*img[srcY1,srcX1,i]
                v2 = (srcY2-srcY)*(srcX-srcX1)*img[srcY1,srcX2,i]
                v3 = (srcY-srcY1)*(srcX2-srcX)*img[srcY2,srcX1,i]
                v4 = (srcY-srcY1)*(srcX-srcX1)*img[srcY2,srcX2,i]
                v=v1+v2+v3+v4
                # 合并的写法  
                #v1 = (srcY2-srcY)*((srcX2-srcX)*img[srcY1,srcX1,i]+(srcX-srcX1)*img[srcY1,srcX2,i])
                #v2 = (srcY-srcY1)*((srcX2-srcX)*img[srcY2,srcX1,i]+(srcX-srcX1)*img[srcY2,srcX2,i])
                #求新图片中 j,k,i点的像素v    
                #v=v1+v2
                newImg[j,k,i]=int(v)
    return newImg

img=cv2.imread("lenna.png")#原图尺寸512*512
#放大后的图片
newImg1=linearInterpolation(img, 800, 800)
#缩小后的图片
newImg2=linearInterpolation(img, 200, 200)

newImg3=cv2.resize(img,dsize=(800,800),interpolation=cv2.INTER_LINEAR)
newImg4=cv2.resize(img,dsize=(200,200),interpolation=cv2.INTER_LINEAR)

cv2.imshow("800*800",newImg1)
cv2.imshow("200*200",newImg2)
cv2.imshow("cv2:800*800",newImg3)
cv2.imshow("cv2:200*200",newImg4)

cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
#                 
# 原理说明：见图 双线性插值公式推导.png
# 使几何中心相等，求出目标点对应的原图的点，坐标为x,y，记为f(x,y)
# 根据向下取整和加1个像素可取到x1,x2,y1,y2
# 从而定位4个点Q11,Q21,Q12,Q22,坐标分别为(x1,y1),(x2,y1),(x1,y2),(x2,y2)
# 点f(x,y)占4个点的比重分别为：
# x1,x2,y1,y2相距都是1，所以以下的分母可以舍掉
# P-Q11(左下角的点比重)=(y2-y)/(y2-y1) * (x2-x)/(x2-x1) * Q11 = (y2-y)(x2-x)Q11
# P-Q21(右下角的点比重)=(y2-y)/(y2-y1) * (x-x1)/(x2-x1) * Q21 = (y2-y)(x-x1)Q21
# P-Q12(左上角的点比重)=(y-y1)/(y2-y1) * (x2-x)/(x2-x1) * Q12 = (y-y1)(x2-x)Q12
# P-Q22(右上角的点比重)=(y-y1)/(y2-y1) * (x-x1)/(x2-x1) * Q22 = (y-y1)(x-x2)Q22
# 四个相加可得 f(x,y)
# =============================================================================
