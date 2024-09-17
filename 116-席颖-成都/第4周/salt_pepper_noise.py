import cv2
import numpy as np
import random

def salt_pepper_noise(img,percentage):
    noiseImg = np.copy(img)
    w,h      = img.shape[0],img.shape[1]
    noisecnt = int(w*h*percentage)
    for i in range(noisecnt):
        randx = random.randint(0,w-1)
        randy = random.randint(0,h-1)
        if random.random()<=0.5:
            noiseImg[randx,randy] = 0
        else:
            noiseImg[randx,randy] = 255
    return noiseImg
img = cv2.imread("lenna.png",0)
img1= salt_pepper_noise(img,0.5)
img = cv2.imread("lenna.png")
img2= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("src",img2)
cv2.imshow("lina_salt_pepper",img1)
cv2.waitKey(0)