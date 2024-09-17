import cv2
import numpy as np

def near(img):
    h,w,ch=img.shape
    empty=np.zeros((1200,1200,ch),np.uint8)
    sh=1200/h
    sw=1200/w
    for i in range(1200):
        for j in range(1200):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            empty[i,j]=img[x,y]
    return empty

img = cv2.imread("2.bmp")
cv2.imshow("123",img)
zoom=near(img)
cv2.imshow("789",zoom)
cv2.waitKey(0)