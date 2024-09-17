import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import util

'''高斯噪声调用API完成的结果'''
# 默认是BGR读取的。
img=cv2.imread('lenna.png')
# 将BGR转换为RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#因为cv2.imread()函数读入的图像默认为BGR格式（即蓝、绿、红顺序），而matplotlib中展示图片默认为RGB格式（即红、绿、蓝顺序）。
noise_gs_img=util.random_noise(img_rgb,mode='gaussian',mean=0,var=0.05)

'''椒盐噪声调用API实现的结果'''
noise_jiaoyan_img=util.random_noise(img_rgb,mode='s&p',amount=0.5,salt_vs_pepper=0.5)

fig,ax=plt.subplots(2,2)

ax[0,0].imshow(img_rgb)
ax[0,1].imshow(noise_gs_img)
ax[1,0].imshow(img_rgb)
ax[1,1].imshow(noise_jiaoyan_img)

plt.show()

