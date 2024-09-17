
import numpy as np
import cv2
from numpy import shape 
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    h, w = NoiseImg.shape[:2]  # 获取图片的high和wide
    NoiseNum=int(percetage * h * w)  # 计算噪声点像素
    for i in range(NoiseNum):
		#取一个随机点
        randX=random.randint(0,h-1)
        randY=random.randint(0,w-1)
        #在原有像素灰度值上加上高斯随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,5,10,0.5)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
