import cv2
import numpy as np

def nearest_interp(img, target_h, target_w): # 参数为需插值的图片与目标像素长宽

    # 储存传进来的图片的长、宽、通道
    h, w, c = img.shape

    # 计算变化的比率
    sh = target_h / h
    sw = target_w / w

    # 创建一个空的矩阵来容纳变换后的图片
    emptyImage = np.zeros([target_h, target_w, c], dtype='uint8')

    # 最邻近插值法原理：
    for i in range(target_h):
        for j in range(target_w):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i, j] = img[x, y]

    return emptyImage

if __name__ == '__main__':

    # # 使用库中已有的函数
    # img = cv2.imread('../lenna.png')
    # img_change = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('lenna', img)
    # cv2.imshow('lenna_change', img_change)
    # cv2.waitKey()
    # 使用自己封装的函数
    img = cv2.imread('../lenna.png')
    img_change = nearest_interp(img, 600, 600)
    cv2.imshow('lenna', img)
    cv2.imshow('lenna-change', img_change)
    cv2.waitKey()

