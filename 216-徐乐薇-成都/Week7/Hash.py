import cv2
import numpy as np

# 均值哈希
def aHash(img):
    # 缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC) #INTER_CUBIC是一种插值算法， 用于放大图像
    # 转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s+=gray[i,j]
    # 求平均灰度
    avg=s/64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

# 差值感知算法
def dHash(img):
    # 缩放8*9
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    # 转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

# Hash值对比
def cmpHash(hash1,hash2):
    n=0
    # hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_blur.png')
img3 = cv2.imread('lenna_color.png')
img4 = cv2.imread('lenna_noise.png')
hash1 = aHash(img1)
hash2 = aHash(img2)
hash3 = aHash(img3)
hash4 = aHash(img4)
print(hash1)
print(hash2)
print(hash3)
print(hash4)
n = cmpHash(hash1,hash2)
m = cmpHash(hash1,hash3)
l = cmpHash(hash1,hash4)
print('lenna与lenna_blur的均值哈希算法相似度：',n)
print('lenna与lenna_color的均值哈希算法相似度：',m)
print('lenna与lenna_noise的均值哈希算法相似度：',l)

hash1 = dHash(img1)
hash2 = dHash(img2)
hash3 = dHash(img3)
hash4 = dHash(img4)
print(hash1)
print(hash2)
print(hash3)
print(hash4)
n = cmpHash(hash1,hash2)
m = cmpHash(hash1,hash3)
l = cmpHash(hash1,hash4)
print('lenna与lenna_blur的差值哈希算法相似度：',n)
print('lenna与lenna_color的差值哈希算法相似度：',m)
print('lenna与lenna_noise的差值哈希算法相似度：',l)
