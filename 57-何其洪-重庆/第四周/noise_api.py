# -*- coding: utf-8 -*-
import cv2
from skimage import util

if __name__ == '__main__':
    original = cv2.imread('../resources/images/lenna.png')
    cv2.imshow('original', original)
    cv2.imshow('img', util.random_noise(original, mode='gaussian'))
    cv2.waitKey(0)
