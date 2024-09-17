# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
#
def gaussion_noise(img,means,sigma,percetage):
    noiseImg = np.copy(img)
    w,h      = img.shape[0],img.shape[1]
    noisecnt = int(w*h*percetage)
    for i in range(noisecnt):
        randx = random.randint(0,w-1)
        randy = random.randint(0,h-1)
        noiseImg[randx,randy] =  noiseImg[randx,randy] + random.gauss(means,sigma)
        if noiseImg[randx,randy] < 0 :
            noiseImg[randx, randy] = 0
        elif noiseImg[randx,randy] > 255:
            noiseImg[randx, randy] = 255
    return noiseImg
img = cv2.imread("lenna.png",0)
img1= gaussion_noise(img,2,5,0.9)
img = cv2.imread("lenna.png")
img2= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("source",img2)
cv2.imshow("lina_guassion_noise",img1)
cv2.waitKey(0)