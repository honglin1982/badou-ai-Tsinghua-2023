import cv2
import numpy as np


# 定义双线插值函数，参数为原图hwc、目标图的hw元组
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w =", src_h, src_w)
    print("dst_h, dst_w =", src_h, src_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.unit8)
    sca_x, sca_y = float(src_w)/dst_w, float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 将原图和目标图的几何中心点对齐
                src_x = (dst_x + 0.5) * sca_x - 0.5
                src_y = (dst_y + 0.5) * sca_y - 0.5
                # 将四个坐标点列出，以x为基准，np.floor是向下取整。x1就是x+1,防止出边界采用两者的min。
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 将公式的两个大块先列出，再组合。避免公式过于繁琐
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
