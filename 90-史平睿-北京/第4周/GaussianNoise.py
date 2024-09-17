
import cv2 as cv
import numpy as np
from numpy import shape
import random

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
		#每次取一个随机点
		#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
		#高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg

img = cv.imread("lenna.png", 0)
img1 = GaussianNoise(img, 2, 4, 0.3)
cv.imshow("lenna_GaussianNoise", img1)

img = cv.imread('lenna.png')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("source", img2)
cv.waitKey(0)