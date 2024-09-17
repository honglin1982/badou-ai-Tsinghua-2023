# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage import util, img_as_float    # 导入所需要的 skimage 库

# mu:平均值；sigma:标准差,用来控制添加噪声程度，sigma越大添加的噪声越多图片损坏的越厉害
#random.gauss(mu, sigma)

def gaussNoise_all(src,mu,sigma):
    targetImg=src
    #行 乘 列，所有的点
    for i in range(src.shape[0]*src.shape[1]):
        #把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
        #排除边缘的点
        x=random.randint(0, src.shape[0]-1)
        y=random.randint(0, src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        targetImg[x,y]=targetImg[x,y]+random.gauss(mu, sigma)
        
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        # if targetImg[x, y]< 0:
        #     targetImg[x, y]=0
        # elif targetImg[x, y]>255:
        #     targetImg[x, y]=255
        
        # np.clip( a, a_min, a_max, out=None)
        # a：输入矩阵；a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；out：可以指定输出矩阵的对象，shape与a相同
        targetImg=np.clip(targetImg,0,255)
    return targetImg

def gaussNoise(src,mu,sigma,percentage):
    targetImg=src
    #行 乘 列，所有的点
    for i in range(int(percentage*src.shape[0]*src.shape[1])):
        #把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
        #排除边缘的点
        x=random.randint(0, src.shape[0]-1)
        y=random.randint(0, src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        targetImg[x,y]=targetImg[x,y]+random.gauss(mu, sigma)
        
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if targetImg[x, y]< 0:
            targetImg[x, y]=0
        elif targetImg[x, y]>255:
            targetImg[x, y]=255
        
        # np.clip( a, a_min, a_max, out=None)
        # a：输入矩阵；a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；out：可以指定输出矩阵的对象，shape与a相同
        #targetImg=np.clip(targetImg,0,255)
    return targetImg

def skimageGaussNoise(src):
    # 转换为 skimage 可操作的格式
    img = img_as_float(src)	
       
    image_gaussian = util.random_noise(img, mode="gaussian")      # 加高斯噪声
    #image_sp = util.random_noise(img, mode="s&p")             # 加椒盐噪声
    return image_gaussian
    
if __name__ == '__main__':
    #orgImg=cv2.imread("lenna.png",0)
    img1=cv2.imread("lenna.png",0)
    target1=gaussNoise_all(img1,2,40)
    
    img2=cv2.imread("lenna.png",0)
    target2=gaussNoise(img2,2,40,0.5)
    # cv2.imshow('org',orgImg)
    # cv2.imshow('gaussNoise_all',target1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #skimage库
    img3=cv2.imread("lenna.png",0)
    target3=skimageGaussNoise(img3)
    
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    plt.figure(figsize=(10, 10))
    img_plt = plt.imread("lenna.png") 
    img_gray = rgb2gray(img_plt)
    plt.subplot(321),plt.title("原图")
    plt.imshow(img_gray, cmap="gray")
    
    gaussAll = Image.fromarray(cv2.cvtColor(target1, cv2.COLOR_BGR2RGB))
    plt.subplot(322),plt.title("全部高斯")
    plt.imshow(gaussAll, cmap="gray")
    
    gauss = Image.fromarray(cv2.cvtColor(target2, cv2.COLOR_BGR2RGB))
    plt.subplot(323),plt.title("部分高斯")
    plt.imshow(gauss, cmap="gray")
    
    #gaussSim = Image.fromarray(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.subplot(324),plt.title("skimage实现高斯")
    plt.imshow(target3, cmap="gray")
    
    plt.show()