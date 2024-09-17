import cv2
import numpy as np
import math


'''
实现最近邻插值
实现双线性插值
证明几何中心对称系数
实现直方图均衡化
'''

'''
证明几何中心对称系数如下:
    原点重合时，原图上一个点(src_x, src_y)到原图左边缘的距离 在原图总宽度的占比 等于 目标图对应点(dst_x, dst_y)到目标图左边缘的距离 在目标图总宽度的占比
    因为以左上角为原点，所以到图左边缘的距离即横坐标，
        即 src_x / src_width = dst_x / dst_width，
        变形后即 src_x = dst_x * src_width / dst_width
    
    同样，当几何中心重合时，原图上一个点(src_x, src_y)到原图中心的水平距离 在原图总宽度的占比 等于 目标图对应点(dst_x, dst_y)到目标图中心的水平距离 在目标图总宽度的占比
    原图上一个点(src_x, src_y)到原图中心的水平距离是 src_x - (src_width - 1) / 2
        即 (src_x - (src_width - 1) / 2) / src_width = (dst_x - (dst_width - 1) / 2) / dst_width
        变形后即 src_x + 0.5 = (dst_x + 0.5) * src_width / dst_width
    
    y方向同理
'''

def diy_interp(img, dst_shape, func="nearest"):
    src_h, src_w, channel = img.shape
    dst_w, dst_h = dst_shape
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    for channel_index in range(channel):
        for i in range(dst_h):
            for j in range(dst_w):
                if func == "nearest":   # 最邻近插值
                    src_x = int(j * src_w / dst_w + 0.5)
                    src_y = int(i * src_h / dst_h + 0.5)
                    dst_img[i, j, channel_index] = img[src_y, src_x, channel_index]
                elif func == "bilinear":    # 双线性插值
                    # 虚拟点的坐标
                    src_x = (j + 0.5) * src_w / dst_w - 0.5
                    src_y = (i + 0.5) * src_h / dst_h - 0.5

                    # 虚拟点附件的四个点的坐标
                    src_x0 = int(math.floor(src_x))
                    src_y0 = int(math.floor(src_y))
                    src_x1 = min(src_x0 + 1, src_w - 1)
                    src_y1 = min(src_y0 + 1, src_h - 1)

                    interpolation_y0 = (src_x1 - src_x) * img[src_y0, src_x0, channel_index] + (src_x - src_x0) * img[src_y0, src_x1, channel_index]
                    interpolation_y1 = (src_x1 - src_x) * img[src_y1, src_x0, channel_index] + (src_x - src_x0) * img[src_y1, src_x1, channel_index]
                    interpolation = (src_y1 - src_y) * interpolation_y0 + (src_y - src_y0) * interpolation_y1

                    dst_img[i, j, channel_index] = int(interpolation)
    return dst_img


img = cv2.imread("lenna.png")
dst_nearest_img = diy_interp(img, (750, 800), func="nearest")
dst_bilinear_img = diy_interp(img, (750, 800), func="bilinear")
cv2.imshow("img", img)
cv2.imshow("dst_nearest_img", dst_nearest_img)
cv2.imshow("dst_bilinear_img", dst_bilinear_img)
cv2.waitKey()

print(dst_nearest_img == dst_bilinear_img)