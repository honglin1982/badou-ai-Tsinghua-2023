import cv2
import random

def fun(src,percetage):
    noise_img = src.copy()
    noise_num = int(percetage * noise_img.shape[0] * noise_img.shape[1])
    for i in range(noise_num):
        randX = random.randint(0,noise_img.shape[0]-1)
        randY = random.randint(0,noise_img.shape[1]-1)

        if random.random() < 0.5 :
            noise_img[randX,randY] = 0
        else:
            noise_img[randX,randY] = 255
    return noise_img

img_gray = cv2.imread('lenna.png',0)
noise_img = fun(img_gray,0.05)

cv2.imshow('img_gray',img_gray)
cv2.imshow('noise_img',noise_img)
cv2.waitKey(0)
                


