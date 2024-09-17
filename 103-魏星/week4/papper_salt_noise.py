import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

'''
1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
3.随机获取要加噪的每个像素位置P（i, j）
4.指定像素值为255或者0。
5.重复3, 4两个步骤完成所有NP个像素的加噪
'''

def papper_salt_noise(src_img,percent):
    h,w,c = src_img.shape
    result_img = np.copy(src_img)
    pixelNum = int(h*w*percent)
    for i in range(pixelNum):
        for j in range(c):
            x = random.randint(0,h-1)
            y = random.randint(0,w-1)
            if random.random() < 0.5:
                result_img[x,y,j] = 0
            else:
                result_img[x, y, j] = 255
    return result_img

def salt_noise(src_img,percent):
    h,w,c = src_img.shape
    result_img = np.copy(src_img)
    pixelNum = int(h*w*percent)
    for i in range(pixelNum):
        for j in range(c):
            x = random.randint(0,h-1)
            y = random.randint(0,w-1)
            if random.random() > 0.5:
                result_img[x, y, j] = 255

    return result_img


def papper_noise(src_img, percent):
    h, w, c = src_img.shape
    result_img = np.copy(src_img)
    pixelNum = int(h * w * percent)
    for i in range(pixelNum):
        for j in range(c):
            x = random.randint(0, h - 1)
            y = random.randint(0, w - 1)
            if random.random() < 0.5:
                result_img[x, y, j] = 0

    return result_img

img = cv2.imread("flower.png")
img2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

noiseImg = papper_salt_noise(img2,0.8)
noiseImg2 = papper_noise(img2,0.8)
noiseImg3 = salt_noise(img2,0.8)

plt.show()
plt.figure()

plt.subplot(221)
plt.imshow(img2)
plt.title("src img")

plt.subplot(222)
plt.imshow(noiseImg)
plt.title("papper&salt noise img")

plt.subplot(223)
plt.imshow(noiseImg2)
plt.title("papper noise img")

plt.subplot(224)
plt.imshow(noiseImg3)
plt.title("salt noise img")
plt.show()



