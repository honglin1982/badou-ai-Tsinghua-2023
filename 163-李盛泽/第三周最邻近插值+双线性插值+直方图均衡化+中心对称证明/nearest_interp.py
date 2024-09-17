#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from timeit import default_timer as timer

def nearest_interp(img,out_dim):
    src_h,src_w,channels =img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]
    print("src_h, src_w = ",src_h,src_w)
    print("dst_h, dst_w = ",dst_h,dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channels),dtype = np.uint8)
    scale_x, scale_y = (dst_h/src_h), (dst_w/src_w)
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = int(dst_x/scale_x + 0.5)
                src_y = int(dst_y/scale_y + 0.5)
                dst_img[dst_y,dst_x,i] = img[src_y,src_x,i]

    return dst_img



if __name__ == '__main__':
    start = timer()
    img = cv2.imread('lenna.png',1)
    dst = nearest_interp(img,(800,800))
    cv2.imshow('nearest_interp',dst)
    print("with nearest_interp: ", timer() - start)
    cv2.waitKey(0)