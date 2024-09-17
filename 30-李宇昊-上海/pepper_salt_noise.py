import cv2
import random

src = img
def pepper_salt(img,percentage):
    NP = int(src.shape[0]*src.shape[1]*percentage)
    for i in range(NP):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        if random.random() <= 0.5:
            src[randx,randy] = 0
        else:
            src[randx,randy] = 255
    return src


if __name__ == '__main__':
    img = cv2.imread('lenna.jpg')
    img1 = pepper_salt(img,0.2)
    img = cv2.imread('lenna.jpg')
    cv2.imshow('source',img)
    cv2.imshow('result',img1)
    cv2.waitKey(0)