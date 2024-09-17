# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度
img = cv2.imread('lenna.png', 0) 
# print(img.shape) # (512, 512)

rows, cols = img.shape

# 图像二维像素转换为一维
data = img.reshape(-1, 1)
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签，这一句必须有，不然显示不出中文
plt.rcParams['font.sans-serif']=['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']  
images = [img, dst]  
for i in range(2):  
   # plt.subplot(nrows, ncols, index)，均等地划分画布，这里的index初始值为1
   plt.subplot(1, 2, i + 1)
   plt.imshow(images[i], 'gray')
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
