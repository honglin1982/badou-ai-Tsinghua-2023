import numpy as np
import cv2
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmphash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n= n + 1
    return n

# 原图加噪声
# img1 = cv2.imread('lenna.png', 0)
# noise_img = GaussianNoise(img1, 2, 4, 0.8)
# lennanoise = cv2.imwrite('lennanoise.png', noise_img)


# 均值哈希
img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lennanoise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print('hash1:', hash1)
print('hash2:', hash2)
n = cmphash(hash1, hash2)
print('均值哈希算法相似度：', n)
# 差值哈希
hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmphash(hash1, hash2)
print('差值哈希算法相似度：', n)
