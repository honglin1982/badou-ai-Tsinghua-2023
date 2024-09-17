#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/6/14
@author: 81-于海洋
"""

import cv2
import numpy as np

from config import Config


def key_point():
    opencv_version = cv2.__version__
    print("OpenCV 版本:", opencv_version)

    img = cv2.imread(Config.LENNA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    kp2, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(img, kp, img)

    cv2.imshow('sift_kp', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    key_point()
