# 均值哈希算法与差值哈希算法

import cv2
import numpy as np

# 均值哈希算法
def aHash(img):
    # step1. 缩放为8 * 8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)

    # step2. 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # step3. 求像素和
    s = 0
    hash_str = []

    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s/64

    # step4. 像素点与平均值比大小，1为大于
    for i in range(8):
        for j in range(8):
            hash_str += '1' if gray[i,j] > avg else '0'

    return hash_str


# 差值哈希算法
def dHash(img):
    # step1. 缩放8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)

    # step2. 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # step3 求差值
    hash_str = ''
    for i in range(8):
        for j in range(8):
            hash_str += '1' if gray[i, j] > gray[i, j + 1] else '0'
    return hash_str


# 海明距离
def cmpHash(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1

    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

img1 = cv2.imread('source.png')
img2 = cv2.imread('noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n) # 3
 
hash1 = dHash(img1)
hash2 = dHash(img2)
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n) # 2
