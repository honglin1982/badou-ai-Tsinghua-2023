import cv2
import numpy as np
import time

def resize(src,new_size):
    dst_w,dst_h=new_size
    src_h,src_w=src.shape[:2]
    if src_h==dst_h and src_w==dst_w:
        return src.copy()
    scale_x=float(src_w)/dst_w
    scale_y=float(src_h)/dst_h

    dst=np.zeros((dst_h,dst_w,3),dtype=np.unit8)
    for n in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)*scale_x-0.5
                src_y=(dst_y+0.5)*scale_y-0.5

                src_x0=int(np.floor(src_x))
                src_y0=int(np.floor(src_y))
                src_x1=min(src_x0+1,src_w-1)
                src_y1=min(src_y0+1,src_h-1)

                value0=(src_x1-src_x)*src[src_y0,src_x0,n]+(src_x-src_x0)*src[src_y0,src_x1,n]
                value1=(src_x1-src_x)*src[src_y1,src_x0,n]+(src_x-src_x0)*src[src_y1,src_x1,n]
                dst[dst_y,dst_x,n]=int((src_y1-src_y)*value0+(src_y-src_y0)*value1)
        return dst

img=cv2.imread('2.bmp')
img1=cv2.resize(img,(1200,1200))
cv2.imshow('image',img1)
cv2.waitKey(0)