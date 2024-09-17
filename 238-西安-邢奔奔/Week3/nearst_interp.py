#!/usr/bin/python
# -*-coding:utf-8 -*-
import cv2
import numpy as np


def nearst_interp(img1, dst_height, dst_width):
    height, width, channels = img1.shape
    dstimage = np.zeros((dst_height, dst_width, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(dst_width):
        for j in range(dst_height):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            dstimage[i, j] = img1[x, y]
    return dstimage


if __name__ == '__main__':
    img = cv2.imread("/Users/aragaki/artificial/image/lenna.png")
    zoom = nearst_interp(img, 800, 800)
    print(zoom)
    print(zoom.shape)
    cv2.startWindowThread()
    cv2.imshow("nearst interp", zoom)
    cv2.imshow("image", img)
    cv2.waitKey(0)  # 在此之前一直一闪而过，到这才能看到
