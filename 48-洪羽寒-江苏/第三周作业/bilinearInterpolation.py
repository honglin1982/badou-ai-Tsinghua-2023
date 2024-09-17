import time
start_time = time.time()

import numpy as np
import cv2

def bilinear_interpolation(img,out_dim):                        #此处输入即是刚才输出图大小：700*700
    src_h,src_w,channel=img.shape                               #读取长宽通道，
    dst_h,dst_w = out_dim[1],out_dim[0]
    print("src_h,src_w=",src_h,src_w)
    print("dst_h,dst_w=",dst_h,dst_w)
    if src_h == dst_h and src_w == dst_w:                        #自动判断，如果原始图片分辨率要求和目标分辨率相等，则直接copy
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)          #建立空图像矩阵
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):                                          #有3个通道，所以range3
        for dst_y in range (dst_h):
            for dst_x in range (dst_w):

                src_x = (dst_x + 0.5)*scale_x - 0.5             #中心重叠；-0.5 是左边src_x加的给移动到右边了。
                src_y = (dst_y + 0.5)*scale_x - 0.5             #为的就是避免是负数导致加成错误

                src_x0 = int(np.floor(src_x))                   #套公式
                src_x1 = min(src_x0 + 1,src_w - 1 )
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1,src_h - 1 )

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(int(input('请输入您需要的水平像素数:')),int(input('请输入您需要的垂直像素数:'))))                         #原本版本是固定式，已优化需求版可让客户手动输入；当前版本只能将横纵分辨率分开输入...
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()