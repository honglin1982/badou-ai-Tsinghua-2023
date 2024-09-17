'''
1.实现最临近插值 2.实现双线性插值 3.证明几何中心对称系数 4.实现直方图均衡化
作业提交的地址：
https://github.com/michael0420/badou-ai-Tsinghua-2023

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Test:
    # 实现最邻近差值
    def nearest_neighbor(self, img):
        height, width, channels = img.shape
        dstimg = np.zeros((800, 800, channels), np.uint8)
        sh = 800 / height
        sw = 800 / width
        for i in range(800):
            for j in range(800):
                x = int(i / sh + 0.5)
                y = int(j / sw + 0.5)
                dstimg[i, j] = img[x, y]
        return dstimg

    # 实现双线性插值
    def bilinear_interpolation(self, img, out_dim):
        src_h, src_w, channel = img.shape
        dst_h, dst_w = out_dim[1], out_dim[0]
        print("src_h, src_w = ", src_h, src_w)
        print("dst_h, dst_w = ", dst_h, dst_w)
        if src_h == dst_h and src_w == dst_w:
            return img.copy()
        dstimg = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
        for i in range(3):
            for dst_y in range(dst_h):
                for dst_x in range(dst_w):
                    src_x = (dst_x + 0.5) * scale_x - 0.5
                    src_y = (dst_y + 0.5) * scale_y - 0.5

                    src_x0 = int(np.floor(src_x))
                    src_x1 = min(src_x0 + 1, src_w - 1)
                    src_y0 = int(np.floor(src_y))
                    src_y1 = min(src_y0 + 1, src_h - 1)

                    temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                    temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                    dstimg[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
        return dstimg

    # 实现直方图均衡化
    def equali(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dst = cv2.equalizeHist(gray)
        hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

        plt.figure()
        plt.hist(dst.ravel(), 256)
        plt.show()
        cv2.imshow('Histogram Equalization', np.hstack([gray, dst]))


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    picture = Test()

    # 最邻近插值
    neast = picture.nearest_neighbor(img)
    print("+++++++++++++++++++nearset_interpolation++++++++++++++++++++++++++++")
    print(neast)
    print(neast.shape)
    cv2.imshow('nearest interp', neast)
    cv2.imshow('image', img)

    # 双线性插值
    bili = picture.bilinear_interpolation(img, (700, 700))
    print('*********************bilinear_interpolation**************************')
    print(bili)
    cv2.imshow('bilinear interp', bili)

    # 实现直方图均衡化
    picture.equali(img)

    cv2.waitKey(0)
