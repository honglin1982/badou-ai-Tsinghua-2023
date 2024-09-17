import numpy as np
import cv2
from numpy import shape
import random
def  PepperSaltNoise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*NoiseImg.shape[0]*NoiseImg.shape[1])
    for i in range(NoiseNum): 
		#每次取一个随机点
	    randX=random.randint(0,src.shape[0]-1)
	    randY=random.randint(0,src.shape[1]-1)
	    #给随机点随机赋予0或255的灰度值
	    if random.randint(0,1)==0:
	    	NoiseImg[randX,randY]=0       
	    else:            
	    	NoiseImg[randX,randY]=255    
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=PepperSaltNoise(img,0.1)


img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)

