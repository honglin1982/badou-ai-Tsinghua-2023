import cv2
import random


def func(src, means, sigma, perc):
    noise_img = src
    noise_num = int(perc * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        noise_img[randx, randy] += random.gauss(means, sigma)
        if noise_img[randx, randy] < 0:
            noise_img[randx, randy] = 0
        elif noise_img[randx, randy] > 255:
            noise_img[randx, randy] = 255
    return noise_img


img = cv2.imread('lenna.png', 0)
img1 = func(img, 2, 4, 0.99)
cv2.imshow('高斯噪声', img1)
cv2.waitKey(0)
