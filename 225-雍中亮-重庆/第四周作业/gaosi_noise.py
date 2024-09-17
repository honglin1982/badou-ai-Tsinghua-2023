import cv2
import random
import numpy as np
from numpy import shape

def GaussianNoise(img,mean,sigma,percetage):
    noiseimg=img
    noisenum=int(percetage*img.shape[0]*img.shape[1])
    for i in range(noisenum):
        randx=random.randint(0,img.shape[0]-1)
        randy=random.randint(0,img.shape[1]-1)
        noiseimg[randx,randy]=noiseimg[randx,randy]+random.gauss(mean,sigma)
        if noiseimg[randx,randy]<0:
            noiseimg[randx,randy]=0
        elif noiseimg[randx,randy]>255:
            noiseimg[randx,randy]=255
    return noiseimg
img_lenna=cv2.imread("lenna.png",0)
img_new_lenna=GaussianNoise(img_lenna,2,4,0.8)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("source",img2)
cv2.imshow("gaussianNoise_lenna",img_new_lenna)
cv2.waitKey(0)


