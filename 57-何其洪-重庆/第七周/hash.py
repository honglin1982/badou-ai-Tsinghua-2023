# -*- coding: utf-8 -*-
import cv2
from skimage import util


def avg_hash(img):
    """
    均值哈希
    :param img: 图片
    :return: 哈希字符串
    """
    # 1. 缩放：图片缩放为8*8，保留结构，除去细节。
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 2. 灰度化：转换为灰度图。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 求平均值：计算灰度图所有像素的平均值。
    avg = gray.mean()
    # 4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
    # 5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    hash_str = ''
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            if gray[y, x] > avg:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def diff_hash(img):
    """
    差值哈希
    :param img: 图片
    :return: 哈希字符串
    """
    # 1. 缩放：图片缩放为8*9，保留结构，除去细节。
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 2. 灰度化：转换为灰度图。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
    # 4. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    hash_str = ''
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1] - 1):
            if gray[y, x] > gray[y, x + 1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def compareHash(h1, h2):
    n = 0
    if len(h1) != len(h2):
        raise ValueError('参数错误，两个hash值长度不等')
    for i in range(len(h1)):
        if h1[i] != h2[i]:
            n += 1
    return n


if __name__ == '__main__':
    # 原图
    img = cv2.imread('../resources/images/lenna.png')
    # 缩放后的图片
    img2 = cv2.resize(img, (256, 256))
    avg_hash_str = avg_hash(img)
    avg_hash_str2 = avg_hash(img2)
    print("均值哈希比较：", compareHash(avg_hash_str, avg_hash_str2))
    diff_hash_str = diff_hash(img)
    diff_hash_str2 = diff_hash(img2)
    print("差值哈希比较：", compareHash(diff_hash_str, diff_hash_str2))
