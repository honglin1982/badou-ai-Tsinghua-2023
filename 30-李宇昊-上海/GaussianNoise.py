import cv2
import random


def Guassian(img,mean,sigma,percentage):
    N = int(img.shape[0]*img.shape[1]*percentage)
    for i in range(N):
        randX = random.randint(0,img.shape[0]-1)
        randY = random.randint(0,img.shape[1]-1)
        img[randX,randY] = img[randX,randY] + random.gauss(mean,sigma)
        if img[randX, randY].any() > 255:
            img[randX,randY] = 255
        elif img[randX, randY].any() < 0:
            img[randX,randY] = 0
    return img


if __name__ == '__main__':
    img = cv2.imread('lenna.jpg')
    img1 = Guassian(img,2,4,0.5)
    img = cv2.imread('lenna.jpg')
    cv2.imshow('source',img)
    cv2.imshow('result',img1)
    cv2.waitKey(0)