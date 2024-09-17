import numpy as np
import cv2
import matplotlib.pyplot as plt

print('-----简单数组的kmeans-----')
#设置原数组
arr = np.array([[1,3,4,6],[4,3,1,6]])
print('arr:\n',arr)
h,w = arr.shape[0],arr.shape[1]
#转化为一维数组
arr_1d = arr.reshape((-1))
print('arr_1d:\n',arr_1d)
data = np.float32(arr_1d)
print('data:\n',data)
#设置结束条件
criteria = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,10,1.0)
#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
#聚类
compactss,labels,centers = cv2.kmeans(data,2,None,criteria,10,flags)
print('compactss:\n',compactss)
print('labels:\n',labels)
print('centers:\n',centers)
arr1 = labels.reshape((h,w))
print(arr1)

print('-----图像的kmeans-----')
#读取灰度图像
img = cv2.imread('lenna.png',0)

#获取图像的长宽
h,w = img.shape[0],img.shape[1]

#转化为np.float32的一维数组
data = img.reshape((h*w,1))
data = np.float32(data)

#设置聚类结束条件，循环十次，精度1.0
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
#设置质心
flags = cv2.KMEANS_RANDOM_CENTERS
#聚类
compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)
#生成聚类后的图像
dst = labels.reshape((h,w))
#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
#绘制图像
titles = [u'原始图像',u'聚类图像']
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


