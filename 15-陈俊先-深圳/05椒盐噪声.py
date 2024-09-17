import random
import cv2


def func(src, perc):
    noise_img = src
    noise_num = int(perc * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        if random.random() <= 0.5:
            noise_img[randx, randy] = 0
        else:
            noise_img[randx, randy] = 255
    return noise_img


img = cv2.imread('lenna.png')
img1 = func(img, 0.5)
cv2.imshow('椒盐噪声', img1)
cv2.waitKey(0)
