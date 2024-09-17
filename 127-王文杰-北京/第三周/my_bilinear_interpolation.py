import cv2
import numpy

img_src = cv2.imread('lenna.png')
height_src,width_src,channel_src = img_src.shape
height_dst = 800
width_dst = 800
channel_dst = channel_src
img_dst = numpy.zeros((height_dst,width_dst,channel_dst),numpy.uint8)
scale_x = width_src/width_dst
scale_y = height_src/height_dst
for i in range(0,channel_dst):
    for j in range(0,height_dst):
        for k in range(0,width_dst):
            res_x = (k + 0.5)*scale_x - 0.5
            res_y = (j + 0.5)*scale_y - 0.5
            x0 = int(res_x)
            y0 = int(res_y)
            x1 = (x0 + 1) if (x0+1 <= width_src-1) else (width_src-1)
            y1 = (y0 + 1) if (y0+1 <= height_src-1) else (height_src-1)
            tmp0 = (x1-res_x)*img_src[y0,x0,i] + (res_x-x0)*img_src[y0,x1,i]
            tmp1 = (x1-res_x)*img_src[y1,x0,i] + (res_x-x0)*img_src[y1,x1,i]
            img_dst[j,k,i] = int((y1-res_y)*tmp0 + (res_y-y0)*tmp1)
cv2.imshow('img_src',img_src)
cv2.imshow('img_dst',img_dst)
cv2.waitKey(0)