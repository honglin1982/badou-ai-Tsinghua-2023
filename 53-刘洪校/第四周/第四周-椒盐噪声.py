# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:25:16 2023

@author: lhx
"""

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage import util, img_as_float    # 导入所需要的 skimage 库

def saltPepperNoise_all(src):
    targetImg=src
    #行 乘 列，所有的点
    for i in range(src.shape[0]*src.shape[1]):
        #把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
        #排除边缘的点
        x=random.randint(0, src.shape[0]-1)
        y=random.randint(0, src.shape[1]-1)
        #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0      
        if random.random()<=0.5:           
        	targetImg[x,y]=0       
        else:            
            targetImg[x,y]=255    
    return targetImg

def saltPepperNoise(src,percentage):
    targetImg=src
    #行 乘 列，所有的点
    for i in range(int(percentage*src.shape[0]*src.shape[1])):
        #把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
        #排除边缘的点
        x=random.randint(0, src.shape[0]-1)
        y=random.randint(0, src.shape[1]-1)
        #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0      
        if random.random()<=0.5:           
        	targetImg[x,y]=0       
        else:            
            targetImg[x,y]=255   
    return targetImg

def skimageSaltPepperNoise(src):
    # 转换为 skimage 可操作的格式
    img = img_as_float(src)	
    image_sp = util.random_noise(img, mode="s&p")             # 加椒盐噪声
    return image_sp
    
if __name__ == '__main__':
    img1=cv2.imread("lenna.png",0)
    target1=saltPepperNoise_all(img1)
    
    img2=cv2.imread("lenna.png",0)
    target2=saltPepperNoise(img2,0.5)

    #skimage库
    img3=cv2.imread("lenna.png",0)
    target3=skimageSaltPepperNoise(img3)
    
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    plt.figure(figsize=(10, 10))
    img_plt = plt.imread("lenna.png") 
    img_gray = rgb2gray(img_plt)
    plt.subplot(321),plt.title("原图")
    plt.imshow(img_gray, cmap="gray")
    
    imageAll = Image.fromarray(cv2.cvtColor(target1, cv2.COLOR_BGR2RGB))
    plt.subplot(322),plt.title("全部椒盐")
    plt.imshow(imageAll, cmap="gray")
    
    image = Image.fromarray(cv2.cvtColor(target2, cv2.COLOR_BGR2RGB))
    plt.subplot(323),plt.title("部分椒盐")
    plt.imshow(image, cmap="gray")
    
    plt.subplot(324),plt.title("skimage实现椒盐")
    plt.imshow(target3, cmap="gray")
    
    plt.show()