import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

'''
a. 输入参数sigma 和 mean
b. 生成高斯随机数
d. 根据输入像素计算出输出像素
e. 重新将像素值放缩在[0 ~ 255]之间
f. 循环所有像素
g. 输出图像
'''

def gray_img_noise(gray_img,sigma,mean):
    h, w = gray_img.shape
    result = np.zeros((h, w), dtype=gray_img.dtype)
    for i in range(h):
        for j in range(w):
                result[i, j] = gray_img[i, j] + random.gauss(mean, sigma)
                if result[i, j] < 0:
                    result[i, j] = 0
                if result[i, j] > 255:
                    result[i, j] = 255
    return result

'''
马赛克 
将原图像按照比例选取后，对选择的区域加高斯噪声
'''
def gauss_noise_msk(src_img,sigma,mean,startPercent,endPercent):
    h,w,c = src_img.shape
    result_img = np.copy(src_img)
    for i in range(int(h*startPercent),int(h*endPercent)):
        for j in range(int(w*startPercent),int(w*endPercent)):
            for k in range(c):
                result_img[i, j, k] = img[i, j, k] + random.gauss(mean, sigma)
                if result_img[i, j, k] < 0:
                    result_img[i, j, k] = 0
                if result_img[i, j, k] > 255:
                    result_img[i, j, k] = 255
    return result_img



def gauss_noise_img(img,sigma,mean):
    h,w,c = img.shape
    result = np.zeros((h,w,c),dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                result[i,j,k] = img[i,j,k] + random.gauss(mean,sigma)
                if result[i,j,k] < 0:
                    result[i,j,k] = 0
                if result[i,j,k] > 255:
                    result[i,j,k] = 255
    return result

img = cv2.imread("flower.png")
img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

noiseImg = gauss_noise_img(img2,4,2)
noiseImg2 = gauss_noise_msk(img2,4,2,0.3,0.6)
noiseGrayImg = gray_img_noise(img_gray,4,2)

plt.show()
plt.figure()

plt.subplot(231)
plt.imshow(img2)
plt.subplot(232)
plt.imshow(noiseImg)
plt.subplot(233)
plt.imshow(noiseImg2)
plt.subplot(234)
plt.imshow(img_gray)
plt.subplot(235)
plt.imshow(noiseGrayImg)
plt.show()



