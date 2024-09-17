# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:16:36 2023

@author: YYX
"""


import numpy as np
import cv2
from numpy import shape
import random
def fun1(src,percentage):
    NoiseImg=src
    NoiseNum=int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[0]-1)
        #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('G:/test/lenna.png',0)
img1=fun1(img,0.2)
#在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
cv2.imwrite('G:/test/lenna_PepperandSalt.png',img1)

img=cv2.imread('G:/test/lenna.png')