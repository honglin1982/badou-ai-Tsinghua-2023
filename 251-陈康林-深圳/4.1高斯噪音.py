import random
import cv2

def GuassianNoise(src,means,sigma,percetage):
    noise_img = src.copy()
    noise_num = int(percetage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        randX = random.randint(0,noise_img.shape[0]-1)
        randY = random.randint(0,noise_img.shape[1]-1)
        noise_img[randX,randY] = src[randX,randY]+random.gauss(means,sigma)
        if noise_img[randX,randY] < 0:
            noise_img[randX,randY] = 0
        elif noise_img[randX,randY] > 255:
            noise_img[randX,randY] = 255
    return noise_img

img_gray = cv2.imread('lenna.png',0)
img_gauss = GuassianNoise(img_gray ,2,4,0.8)

cv2.imshow('gauss',img_gauss)
cv2.imshow('src',img_gray)
cv2.waitKey()


        



