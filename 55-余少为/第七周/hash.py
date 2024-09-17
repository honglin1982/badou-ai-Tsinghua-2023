import cv2
import numpy as np


def aHash(img):
    # 均值哈希
    img = cv2.resize(img, (8, 8))   # 修改图像尺寸为8*8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    # 计算均值
    avg = np.mean(gray)
    # 比较灰度图与均值，生成该图片的均值哈希指纹
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img):
    # 差值哈希
    img = cv2.resize(img, (9, 8))   # 修改图像尺寸为9*8
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转换为灰度图
    # 计算差值
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmpHash(hash1, hash2):
    # 哈希长度必须一致，否则返回-1
    if len(hash1) != len(hash2):
        return -1

    # 遍历比较，生成明汉距离
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n


img1 = cv2.imread('./source/lenna.png')
img2 = cv2.imread('./source/lenna_sharp.jpg')

hash1 = aHash(img1)
hash2 = aHash(img2)
print('hash1:\n', hash1)
print('hash2:\n', hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = dHash(img1)
hash2 = dHash(img2)
print('hash1:\n', hash1)
print('hash2:\n', hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)
