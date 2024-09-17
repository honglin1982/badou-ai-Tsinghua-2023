import cv2
import numpy as np

#均值hash
def aHash(img):
    #缩放成（8，8）
    img1 = cv2.resize(img,dsize=(8,8),interpolation=cv2.INTER_CUBIC)
    #转成灰度图
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #计算均值
    graysum = 0
    for i in range(0,8):
        for j in range(0,8):
            graysum += gray[i,j]
    avg = graysum/64
    #每一个点与均值比较
    hashvalue = ''
    for i in range(0,8):
        for j in range(0,8):
            if gray[i,j] > avg:
                hashvalue = hashvalue + '1'
            else:
                hashvalue = hashvalue + '0'
    return hashvalue

#差值hash
def dHash(img):
    #缩放成（8，8）
    img1 = cv2.resize(img,dsize=(9,8),interpolation=cv2.INTER_CUBIC)
    #转成灰度图
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #每一个点与均值比较
    hashvalue = ''
    for i in range(0,8):
        for j in range(0,8):
            if gray[i,j] > gray[i,j+1]:
                hashvalue = hashvalue + '1'
            else:
                hashvalue = hashvalue + '0'
    return hashvalue
def cmphash(hashvalue1,hashvalue2):
    if len(hashvalue1) != len(hashvalue2):
        return -1
    
    d=0
    for i in range(len(hashvalue1)):
        if hashvalue1[i] != hashvalue2[i]:
            d += 1
    return d

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
hashvalue1 = aHash(img1)
hashvalue2 = aHash(img2)
d = cmphash(hashvalue1,hashvalue2)
print(hashvalue1)
print(hashvalue2)
print(d)

hashvalue1 = dHash(img1)
hashvalue2 = dHash(img2)
d = cmphash(hashvalue1,hashvalue2)
print(hashvalue1)
print(hashvalue2)
print(d)


