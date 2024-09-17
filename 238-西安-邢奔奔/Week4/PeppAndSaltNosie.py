#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import random
from numpy import shape


def PeppAndSalt(src,percentage):
    NoiseImage = src;
    Noisenum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(Noisenum):
        random_X = random.sample(range(0,src.shape[0]-1),1)
        random_Y = random.sample(range(0,src.shape[1]-1),1)
        if random.random() <= 0.5:
            NoiseImage[random_X,random_Y] = 0
        else:
            NoiseImage[random_X,random_Y] = 255
    return NoiseImage

if __name__ =='__main__':
    img = cv2.imread('/Users/aragaki/artificial/image/lenna.png', 0)
    dstImg = PeppAndSalt(img,0.4)
    cv2.imshow("DstImage",dstImg)
    img = cv2.imread('/Users/aragaki/artificial/image/lenna.png',0)
    cv2.imshow("ContrastImage",img)
    cv2.waitKey(0)
