import cv2
import numpy as np

def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape#读取原图像的高，宽，通道
    dst_h, dst_w = out_dim[1], out_dim[0]#目标图片的尺寸
    if src_h == dst_h and src_w == dst_w:#若原图像与目标图像尺寸相同，则直接copy
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)#初始化全零数组用于存储目标图像
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h#求得比例关系
    for i in range(3):                #三重循环遍历通道，高，宽，分别赋值给目标图像
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5#几何中心重合
                src_y = (dst_y + 0.5) * scale_y - 0.5
                src_x0 = int(np.floor(src_x))#先向下取整在强转
                src_x1 = min(src_x0 + 1, src_w - 1)#设置界限，当超出原图像范围时，设置目标图像像素值为边缘
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                #套用公式，实现双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


img = cv2.imread('lenna.png')
dst = bilinear_interpolation(img,(700,700))
cv2.imshow('bilinear interp', dst)
cv2.waitKey()
