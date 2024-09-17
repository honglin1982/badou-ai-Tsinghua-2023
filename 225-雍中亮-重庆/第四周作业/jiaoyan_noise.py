import numpy as np
import cv2
from numpy import shape
import random

def jiaoyan(src,percetage):
    jiaoyan_img=src
    jiaoyan_num=int(src.shape[0]*src.shape[1]*percetage)
    for i in range(jiaoyan_num):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        if random.random()<=0.5:
            jiaoyan_img[randX,randY]=0
        else:
            jiaoyan_img[randX,randY]=255
    return jiaoyan_img

img=cv2.imread("lenna.png",0)
noise_img=jiaoyan(img,0.5)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("source",img2)
cv2.imshow("jiaoyan_noise",noise_img)
cv2.waitKey(0)

