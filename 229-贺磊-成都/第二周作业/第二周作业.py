# -*- coding: utf-8 -*-
# File  : 第二周作业.py
# Author: HeLei
# Date  : 2023/4/27


from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

"""灰度图实现"""


# 手动灰度化
def hand_gray():
    img = cv2.imread("./data/cat.jpg")  # 默认是BGR模式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为RGB模式
    # img = plt.imread("./data/cat.jpg") #plt方式读图
    h, w = img.shape[:2]  # 读处长和宽
    img_gray = np.zeros([h, w], img.dtype)  # 初始化一个和图片大小一样的0矩阵(灰度图矩阵)
    for i in range(h):
        for j in range(w):
            m = img[i, j]  # 此处m为三通道，有三个值
            img_gray[i, j] = int(m[0] * 0.3 + m[1] * 0.59 + m[2] * 0.11)  # 按照权值分配给灰度图矩阵赋值
    print(img_gray)  # 打印处理后的灰度图
    cv2.imshow('gray picture ', img_gray)  # 展示灰度图
    cv2.waitKey(0)  # 让图像不会一闪而过


# 调用库函数是实现灰度化
def auto_gray():
    # skimage实现灰度化
    img = cv2.imread("./data/cat.jpg")  # 读图
    img_gray = rgb2gray(img)
    cv2.imshow('gray picture ', img_gray)  # 展示灰度图
    print("skimage_image_gray:", img_gray)  # 打印一下灰度图
    cv2.waitKey(0)  # 让图像不会一闪而过

    plt.subplot(221)  # 表示创建2*2表格，当前图显示在第一格，第一个参数代表子图的行数，第二个参数代表该行图像的列数，第三个参数代表每行的第几个图像

    # 使用opencv实现灰度化
    img = cv2.imread("./data/cat.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # 用opencv输出
    cv2.imshow('gray picture ', img_gray)
    cv2.waitKey(0)

    # 用plt输出：这里要注意，cmap为颜色图谱，默认为RGB(A)颜色空间，也可以指定，gray是灰度图，若cmap为其他会造成色差
    plt.imshow(img_gray, cmap="gray")
    plt.show()
    print("opencv_image_gray:", img_gray)  # 打印一下灰度图
    plt.subplot(222)


"""二值化图实现"""


# 手动二值化
def hand_binary():
    """使用plt读图处理图，"""
    # img = plt.imread("./data/lenna.png")
    # img_gray = rgb2gray(img) #首先还是要灰度化
    # rows,cols = img_gray.shape #获取行数,列数
    # for i in range(rows):
    #     for j in range(cols):
    #         if(img_gray[i,j]>=0.5): #大于0.5给白色
    #             img_gray[i,j] = 1
    #         else:                   #小于0.5给黑色
    #             img_gray[i,j] = 0
    # plt.imshow(img_gray,cmap="gray")
    # plt.show()

    """opencv方式"""
    img = cv2.imread('./data/cat.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    rows, cols = img_gray.shape[0:2]
    thresh = 60  # 设置阈值
    for row in range(rows):
        for col in range(cols):
            gray = img_gray[row, col]  # 获取到灰度值
            if gray > thresh:
                img_gray[row, col] = 255  # 如果灰度值高于阈值 就等于255最大值
            elif gray < thresh:
                img_gray[row, col] = 0  # 如果小于阈值，就直接改为0
    cv2.imshow('img', img_gray)
    cv2.waitKey()

    """使用Image读图处理图"""
    # img = Image.open("./data/cat.jpg")
    # Img = img.convert('L') #L表示灰度
    # Img.show() #展示图象
    # threshold = 150 #设置一个阈值
    # table = []
    # for i in range(256):
    #     if i < threshold:
    #         table.append(0)
    #     else:
    #         table.append(1)
    # img1 = Img.point(table, '1') #1表示二值图
    # img1.show()


# 调用库函数实现二值化
def auto_binary():
    """plt实现"""
    # img = plt.imread("./data/cat.jpg")
    # img_gray = rgb2gray(img)  # 灰度化
    # img_binary = np.where(img_gray >= 0.5, 1, 0)
    # plt.imshow(img_binary, cmap='gray')
    # plt.show()

    """"opencv实现"""
    img = cv2.imread("./data/cat.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.threshold(img_gray, 127, 255, 0, img_gray)
    cv2.imshow('img_binary', img_gray)
    cv2.waitKey(0)


if __name__ == '__main__':
    # hand_gray()  # 手动灰度化
    # auto_gray()  # 调用库函数实现灰度化
    # hand_binary()  # 手动二值化
    auto_binary()  # 调用库函数实现二值化
