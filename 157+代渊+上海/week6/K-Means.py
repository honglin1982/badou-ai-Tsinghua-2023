# coding: utf-8

import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

#实现kmeans函数
def kmeans(data, k):
   data_size = data.shape[0]
   print(data_size)
   labels=np.empty(data_size,dtype=int)
   means=np.empty(data_size,dtype=int)
   centers=np.empty(k,dtype=float)
   centers_new=np.empty(k,dtype=float)
   distance = np.empty([data_size,k],dtype=int)
   cluster = [[] for _ in range(k)]
   flag = 0
   for i in range(k):
      centers[i] = data[random.randint(0,data_size-1)]
   print(centers)
   while flag == 0:
      for i in range(data_size):
         for j in range(k):
            distance[i,j] = abs(data[i] - centers[j])
         for j in range(k):
            if distance[i,j] == min(distance[i,:k]):
               cluster[j].append(data[i])
               labels[i] = j
      print(distance)
      for i in range(k):
         if len(cluster[i]) == 0:
            centers_new[i] = 0
         else:
            centers_new[i] = sum(cluster[i])/len(cluster[i])
      if centers.all() == centers_new.all():
         flag = 1
      else:
         centers = centers_new
         del cluster

   for i in range(data_size):
      for j in range(k):
         if labels[i] == j:
            means[i] = centers_new[j]

   return [labels,centers,means]



#读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0) 
print(img.shape)

#获取图像高度、宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)
print(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
print(criteria)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
print(flags)

#K-Means聚类 聚集成4类
labels, centers, means = kmeans(data, 4)

#生成最终图像
dst = means.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']  
images = [img, dst]  
for i in range(2):  
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
