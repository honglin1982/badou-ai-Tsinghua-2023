#实现最邻近插值算法，将512*512转换为800*800
import cv2
import numpy as np

def function(img,a,b):
    height,width,channels =img.shape  #读取原图像高，宽，通道
    emptyImage=np.zeros((a,b,channels),np.uint8)  #初始化全零二维数组，unit8数据类型（0~25）
    sh=a/height  #放大的比例
    sw=b/width
    for i in range(a):
        for j in range(b):
            x=int(i/sh + 0.5)  #i/sh：判断i在哪个区域，+0.5：将四舍五入转换为向上取整
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]  #注意：如转换的图像大小过大时，x，y的取值可能越界，会报错
    return emptyImage

img=cv2.imread("lenna.png")  #读取图片
zoom=function(img,800,800)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)  #最邻近插值后的图片
cv2.imshow("image",img)  #原图
cv2.waitKey(0)
