import numpy as np
import cv2
from numba import jit
import random
import math

@jit(nopython=True)
def GaussNoises(img, percentage, sigma, means):

    h, w, c = img.shape
    Noises = img.copy()
    #算出噪点数
    NoisesNum = math.ceil(h * w * 1.0 * percentage)
    for i in range(c):
        for _ in range(NoisesNum):

            randY, randX = random.randint(0, h - 1), random.randint(0, w - 1)
            #算出噪点值
            pout = img[randY, randX, i] + random.gauss(sigma, means)
            # 写入随机点
            Noises[randY, randX, i] = 0 if pout < 0 else 255 if pout > 255 else pout
    return Noises

@jit(nopython=True)
def PepperAndSaltNoises(img, percentage):

    h, w, c = img.shape
    Noises = img.copy()
    NoisesNum = math.ceil(h * w * 1.0 * percentage)
    for i in range(c):
        for _ in range(NoisesNum):
            randY, randX = random.randint(0, h - 1), random.randint(0, w - 1)
            #根据随机数 写入随机点噪声
            Noises[randY, randX, i] = 0 if random.random() >= 0.5 else 255
    return Noises

img = cv2.imread("lenna.png")
gaussNoises = GaussNoises(img, 1, 18, 2)
pepperAndSaltNoises = PepperAndSaltNoises(img, 0.2)
cv2.imshow("source img", img)
cv2.imshow("GaussNoises img", gaussNoises)
cv2.imshow("PepperAndSaltNoises img", pepperAndSaltNoises)

cv2.waitKey(0)