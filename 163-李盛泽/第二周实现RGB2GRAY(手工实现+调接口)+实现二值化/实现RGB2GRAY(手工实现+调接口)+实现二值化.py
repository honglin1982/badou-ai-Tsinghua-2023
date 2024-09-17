# -*- coding: utf-8 -*-
'''
    @author:shengzeli
    第二周作业：
        1.实现RGB2GRAY(手工实现+调接口)
        2.实现二值化
'''
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = plt.imread("lenna.png")
# 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
plt.subplot(321)
plt.title("1.原图"),plt.axis('off')
plt.imshow(img)

# 实现RGB2GARY
img1 = cv2.imread("lenna.png") #此处注意用plt.imread会自动归一化处理，无法继续用0.3R+0.59G+0.11B公式
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
h,w = img1.shape[:2]
img_gray1 = np.zeros([h,w],img1.dtype)
for i in range(h):
    for j in range(w):
        m = img1[i,j]
        img_gray1[i,j] = int(m[0]*0.3+m[1]*0.59+m[2]*0.11)  # RGB2GRAY
# print(img_gray1)
plt.subplot(322)
plt.title("2.手动"),plt.axis('off')
plt.imshow(img_gray1,cmap = 'gray')

# 调接口
img_gray2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.subplot(323)
plt.title("3.调cv2.cvtColor"),plt.axis('off')
plt.imshow(img_gray2,cmap = 'gray')

img_gray3 = rgb2gray(img)
plt.subplot(324)
plt.title("4.调skimage.color"),plt.axis('off')
plt.imshow(img_gray3,cmap = 'gray')

#二值化
img_binary1 = img_gray3.copy()
r,c = img_binary1.shape
for i in range(r):
    for j in range(c):
        if(img_binary1[i,j] <= 0.5):
            img_binary1[i,j] = 0
        else:
            img_binary1[i,j] = 1
plt.subplot(325)
plt.title("5.手动二值化"),plt.axis('off')
plt.imshow(img_binary1,cmap = 'gray')

img_binary2 = np.where(img_gray3 >= 0.5,1,0)
plt.subplot(326)
plt.title("6.where函数实现二值化"),plt.axis('off')
plt.imshow(img_binary2,cmap = 'gray')

plt.show()

