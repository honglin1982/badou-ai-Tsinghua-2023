import numpy as np
import cv2


def bilinear(img, dst_arr):
    src_h, src_w, channel = img.shape  #提前原图像的高宽，通道
    dst_h, dst_w = dst_arr[1], dst_arr[0]; #变换后图像的高宽
    print("src.shape", src_h, src_w)
    print("dst.shape", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype = np.uint8)
    mul_h, mul_w = float(src_h) / dst_h, float(src_w) / dst_w
    for i in range(3):
        for P_x in range(dst_h):
            for P_y in range(dst_w):

             # 计算出转换后图像像素对应原图像的坐标
             src_x = (P_x + 0.5) * mul_h - 0.5
             src_y = (P_y + 0.5) * mul_w - 0.5

             #目标点周围四个点的范围
             src_x1 = int(np.floor(src_x))
             src_x2 = min(src_x1 + 1,src_w-1)
             src_y1 = int(np.floor(src_y))
             src_y2 = min(src_y1 + 1,src_h-1)

             Q1 = (src_x2-src_x) * img[src_y1,src_x1,i] + (src_x-src_x1) * img[src_y1,src_x2,i]
             Q2 = (src_x2-src_x) * img[src_y2,src_x1,i] + (src_x-src_x1) * img[src_y2,src_x2,i]
             dst_img[P_y,P_x,i] = int((src_y2-src_y) * Q1 + (src_y-src_y1) * Q2)

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear(img, (700,700))
    cv2.imshow('src img', img)
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
