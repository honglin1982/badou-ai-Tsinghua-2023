import cv2
import numpy

img_src = cv2.imread('lenna.png')
height_src,width_src,channel_src = img_src.shape
height_dst = 800
width_dst = 800
channel_dst = channel_src
img_dst = numpy.zeros((height_dst,width_dst,channel_dst),numpy.uint8)
height_scale = height_dst/height_src
width_scale = width_dst/width_src
for i in range(0,height_dst):
    for j in range(0,width_dst):
        i_res = int(i/height_scale+0.5)
        j_res = int(j/width_scale+0.5)
        img_dst[i,j] = img_src[i_res,j_res]
cv2.imshow('img_src', img_src)
cv2.imshow('img_dst', img_dst)
cv2.waitKey(0)
