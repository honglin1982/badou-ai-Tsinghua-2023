'''
canny边缘检测，优化程序
'''

import numpy as np
import cv2

def canny_threshold(low_threshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0) # 高斯滤波
    detected_edges = cv2.Canny(detected_edges,
                               low_threshold,
                               low_threshold * ratio,
                               apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny_demo', dst)


if __name__ == '__main__':
    low_threshold = 0
    max_low_threshold = 100
    ratio = 3
    kernel_size = 3

    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('canny_demo')


    # 设置调节杠
    cv2.createTrackbar('min_threshold',
                       'canny_demo',
                       low_threshold,
                       max_low_threshold,
                       canny_threshold)

    canny_threshold(0)
    if cv2.waitKey(0) == 23:
        cv2.destroyAllWindows()
