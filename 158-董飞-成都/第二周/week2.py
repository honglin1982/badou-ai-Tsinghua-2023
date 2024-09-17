# -*- coding=utf-8 -*-
import cv2
import numpy as np

def my_bgr2gray(img: np.array = None) -> np.array:
    """
    parameter :
        img:输入原始图像
    return :
        输出灰度图
    """
    img_gray = np.zeros(img.shape[:2])  # 根据原图大小创建空白花板
    img_gray = img[:, :, 0]*0.11+img[:, :, 1] * \
        0.59+img[:, :, 2]*0.3  # 三个通道乘以权值得到灰度值
    return img_gray.astype(np.uint8)


def my_binarization(img: np.array = None, threshold: float = 0.5) -> np.array:
    """
    parameter :
        img:输入灰度
    return :
        输出二值图
    """
    img[img > int(255*threshold)] = 255  # 阈值之上变为255
    img[img <= int(255*threshold)] = 0  # 阈值之下变为0
    # img=np.where(img>int(255*threshold),255,0).astype(np.uint8)   
    return img

if "__main__" == __name__:
    img = cv2.imdecode(np.fromfile('1.jpg', dtype=np.uint8), 1)  # 读图
    
    img_gray = my_bgr2gray(img)  # 调用自己写的函数将原图转为灰度图
    cv2.imshow('my_gray', img_gray)
    cv2.waitKey(-1)

    cv2.imshow('cv2_gray', cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY))  # 调用opencv接口函数将原图转为灰度图
    cv2.waitKey(-1)


    img_binarization = my_binarization(img_gray.copy(), 0.6)  # 调用自己写的函数将原图转为二值图
    cv2.imshow('my_binarization', img_binarization)
    cv2.waitKey(-1)

    cv2.imshow('cv2_binarization', cv2.threshold(img_gray, int(
        255*0.5), 255, cv2.THRESH_BINARY)[1])  # 调用opencv接口函数将原图转为二值图
    cv2.waitKey(-1)
