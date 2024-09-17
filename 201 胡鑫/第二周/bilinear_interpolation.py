import cv2
import numpy as np

def bilinear_interpolation(img, dst_h, dst_w):

    # 储存img的长、宽、通道
    src_h, src_w, c = img.shape

    # 计算比率
    sh = src_h / dst_h
    sw = src_w / dst_w

    # 简单判断，若尺寸不发生变化，则退出
    if sh == 1 and sw == 1:
        return img.copy()

    # 创建一个新的矩阵以容纳更改后的图像
    emptyImage = np.zeros([dst_h, dst_w, c], dtype='uint8')

    # 双线性插值原理：
    for channels in range(c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w): # 方便套入公式，dst_y 对应 h，相当于坐标系中y方向，x方向同理

                # 重合几何中心后每个目标坐标系中的位置在原坐标系中的坐标
                src_x = (dst_x + 0.5) * sw - 0.5
                src_y = (dst_y + 0.5) * sh - 0.5

                # 找到下述四个值，就能拼凑出目标像素在原坐标系中四周最邻近的四个坐标，然后进行加权求和
                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1, src_w - 1) # 边界处理，若src_x0 + 1 > src_h - 1, 会运行时异常
                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1, src_h - 1) # 边界处理

                # 双线性插值原理
                # x方向插值
                r1px = (src_x1 - src_x) * img[src_y0, src_x0, channels] + (src_x - src_x0) * img[src_y0, src_x1, channels]
                r2px = (src_x1 - src_x) * img[src_y1, src_x0, channels] + (src_x - src_x0) * img[src_y1, src_x1, channels]
                # y方向插值
                emptyImage[dst_y, dst_x, channels] = int((src_y1 - src_y) * r1px + (src_y - src_y0) * r2px)

    return emptyImage

if __name__ == '__main__':

    # # 使用库中自带的函数
    # img = cv2.imread('../lenna.png')
    # dst_img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('lenna', img)
    # cv2.imshow('dst-lenna', dst_img)
    # cv2.waitKey()

    # 使用自己封装的函数
    img = cv2.imread('../lenna.png')
    dst_img = bilinear_interpolation(img, 600, 600)
    cv2.imshow('lenna', img)
    cv2.imshow('dst-lenna', dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()