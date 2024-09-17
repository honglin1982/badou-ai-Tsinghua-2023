import numpy as np
import cv2
import matplotlib as plt

# 双线性插值
def bilinear_interpolation(img, dst_size):
    # 获取源、目标图像的属性
    src_height, src_width, src_channels = img.shape
    dst_height, dst_width, dst_channels = dst_size
    print("源图像属性：", img.shape)
    print("目标图像属性：", dst_size)

    # 创建目标图像
    dst_img = np.zeros(dst_size, img.dtype)

    # 循环目标图像，通过双线性插值进行赋值
    for dst_c in range(dst_channels):
        for dst_y in range(dst_height):
            for dst_x in range(dst_width):
                # 1、通过中心对称公式求出 目标图像的像素点在源图像上的对应像素点
                src_x = (dst_x + 0.5) * (float(src_width) / dst_width) - 0.5
                src_y = (dst_y + 0.5) * (float(src_height) / dst_height) - 0.5
                # print(src_x, src_y, dst_x, dst_y)
                # 通过观察可知目标图像对应的源图像坐标大概率为小数

                # 2、求距离 src_x,src_y 坐标最近的四个点的坐标，由于双线性插值只会用相邻的4个点，所以：
                x1 = int(np.floor(src_x))
                x2 = x1 + 1
                y1 = int(np.floor(src_y))
                y2 = y1 + 1

                # 3、需要确认求得的4个点不超过源图像的范围
                x1 = max(0, min(x1, src_width - 1))
                x2 = max(0, min(x2, src_width - 1))
                y1 = max(0, min(y1, src_height - 1))
                y2 = max(0, min(y2, src_height - 1))
                # 这四个点构成了一个矩形区域，其中 (x1, y1) 是左下角的点，(x2, y2) 是右上角的点

                # 4、根据双线性插值公式可得
                r1 = (x2 - src_x) * img[y1, x1, dst_c] + (src_x - x1) * img[y1, x2, dst_c]
                r2 = (x2 - src_x) * img[y2, x1, dst_c] + (src_x - x1) * img[y2, x2, dst_c]
                dst_img[dst_y, dst_x, dst_c] = int((y2 - src_y) * r1 + (src_y - y1) * r2)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst_size = (200, 200, 3)
    dst_img = bilinear_interpolation(img, dst_size)
    cv2.imshow("bilinear interpolation", dst_img)
    cv2.waitKey(0)