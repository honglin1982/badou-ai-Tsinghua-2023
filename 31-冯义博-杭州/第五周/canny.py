import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
# 灰度化
g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯函数标准差 这个值越大滤波越平均 越小 中心值越大
sigma = 0.5
# 高斯函数是钟形曲线 窗口越大 影响就越小 6*sigma的窗户大小正合适
size = int(sigma * 6)
# 窗口为5*5效果最好
if size < 5:
    size = 5
elif size % 2 == 0:
    size += 1

gauss_filter = np.zeros((size, size))
tmp = [i - size//2 for i in range(size)]
n1 = 1 / 2 * math.pi * sigma ** 2
n2 = -1 / 2 * sigma ** 2
for i in range(size):
    for j in range(size):
        gauss_filter[i, j] = n1 * math.exp((tmp[i] ** 2 + tmp[j] ** 2) * n2)

gauss_filter = gauss_filter / gauss_filter.sum()
# 边缘填充 高斯滤波 降噪
h, w = g_img.shape[:2]
f_img = np.zeros((w, h))
p = size // 2
n_img = np.pad(g_img, [(p, p), (p, p)])
for i in range(w):
    for j in range(h):
        f_img[i, j] = (n_img[i:i+size, j:j+size] * gauss_filter).sum()

plt.figure(1)
# 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.imshow(f_img, cmap='gray')
plt.axis('off')
# 用sobel分别对x,y方向做卷积 检测图像边缘
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_img_x = np.zeros(f_img.shape)
sobel_img_y = np.zeros(f_img.shape)
sobel_img = np.zeros(f_img.shape)
s_img = np.pad(f_img, ((1, 1), (1, 1)))
h, w = f_img.shape[:2]
for i in range(w):
    for j in range(h):
        sobel_img_x[i, j] = (s_img[i:i+3, j:j+3] * sobel_kernel_x).sum()
        sobel_img_y[i, j] = (s_img[i:i+3, j:j+3] * sobel_kernel_y).sum()
        sobel_img[i, j] = np.sqrt(sobel_img_x[i, j]**2 + sobel_img_y[i, j]**2)

sobel_img[sobel_img_x == 0] = 0.00000001
angle = sobel_img_y/sobel_img_x
plt.figure(2)
plt.imshow(sobel_img.astype(np.uint8), cmap='gray')
plt.axis('off')

img_yizhi = np.zeros(sobel_img.shape)
for i in range(1, w - 1):
    for j in range(1, h - 1):
        flag = True
        temp = sobel_img[i-1:i+2, j-1:j+2]
        # 双线性插值法 x y 方向插值 判断抑制与否
        if angle[i, j] <= -1:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (sobel_img[i, j] > num_1 and sobel_img[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (sobel_img[i, j] > num_1 and sobel_img[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (sobel_img[i, j] > num_1 and sobel_img[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (sobel_img[i, j] > num_1 and sobel_img[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = sobel_img[i, j]

plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 双阈值检测
lower_boundary = img_yizhi.mean()
high_boundary = lower_boundary * 3
h, w = img_yizhi.shape[:2]
zhan = []
# 外圈不处理
for i in range(1, w-1):
    for j in range(1, h-1):
        if img_yizhi[i, j] >= high_boundary:
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        if img_yizhi[i, j] <= lower_boundary:
            img_yizhi[i, j] = 0

plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')


# 连接边缘检测
while not len(zhan) == 0:
    x, y = zhan.pop()
    a = img_yizhi[x-1:x+2, y-1:y+2]
    if(a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[x-1, y-1] = 255
        zhan.append([x-1, y-1])
    if(a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[x-1, y] = 255
        zhan.append([x-1, y])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[x-1, y+1] = 255
        zhan.append([x-1, y+1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[x, y-1] = 255
        zhan.append([x, y-1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[x, y+1] = 255
        zhan.append([x, y+1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[x+1, y-1] = 255
        zhan.append([x+1, y-1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[x+1, y] = 255
        zhan.append([x+1, y])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[x+1, y+1] = 255
        zhan.append([x+1, y+1])

for i in range(w):
    for j in range(h):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

plt.figure(5)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()







