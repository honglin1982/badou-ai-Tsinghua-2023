
import cv2
import numpy as np

img1 = cv2.imread("lenna.png")
h, w = img1.shape[:2]
img1_gray = np.zeros([h, w], img1.dtype)
for i in range(h):
    for j in range(w):
        m = img1[i, j]
        img1_gray[i, j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print(img1_gray)
print("image show gray: %s" % img1_gray)
cv2.imshow("image show gray", img1_gray)
cv2.waitKey()


# 个人笔记

# 逻辑顺序：将原图坐标化，建一张与原图等大小的空白图，将原图rgb数值转为灰度值后赋值到空白图的坐标上（坐标与原图一一对应）

# 因转行基础较差，补充自查细节：

# 读取图片：cv2.imread("在与py文件在同一文件夹时的文件名")
# 获得图片的长 宽：h, w = img.shape[:2]
# 获得图片的长、宽、通道数：h, w, c = img.shape[:3]
# img.shape[0,1,2]： 0是垂直尺寸（高），1是水平尺寸（宽），2是通道数

# np.zeros(shape,dtype=float,order='C')：
# dtype是数据类型，默认是numpy.float64   order是可选参数，C代表行优先，F代表列优先。这里用img1的dtype传入此参数。

# 图像结果在哪看？？？？？

# 用opencv封装好的函数直接将图片灰度化
# img1_gray = rgb2gray(img)
