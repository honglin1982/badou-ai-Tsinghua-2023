import cv2
import numpy
import time
import os.path as path


def aHash(img, width=8, high=8):    # 如果没有提供 width 参数的值，它将默认为 8
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s += gray[i, j]
    avg = s/width*high
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img, width=9, high=8):
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(high):
        for j in range(high):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmp_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1    # 当函数执行到 return 语句时，它会立即退出函数，并将指定的值作为函数的返回值返回给调用者
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return 1 - n/len(hash1)


img = cv2.imread('lenna.png')
img2 = cv2.imread('iphone1.png')
hash1 = aHash(img)
hash2 = aHash(img2)
p = cmp_hash(hash1, hash2)
print(p)
print(hash1)
