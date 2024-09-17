

import cv2 as cv
import numpy as np

def function(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels),np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv.imread('lenna.png')
zoom = function(img)
print(zoom)
print(zoom.shape)
cv.imshow('nearest interp', zoom)
cv.imshow('image', img)
cv.waitKey(0)
