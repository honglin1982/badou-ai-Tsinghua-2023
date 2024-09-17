import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
cv2 kmeans：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
参数：
* data表示聚类数据，最好是np.flloat32类型的N维点集
* K表示聚类类簇数
* bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
* criteria表示算法终止条件，即最大迭代次数或所需精度。在某些迭代中，一旦每个簇中心的移动小于criteria.epsilon，算法就会停止
* attempts表示重复试验kmeans算法的次数，算法返回产生最佳紧凑性的标签
* flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
* centers表示集群中心的输出矩阵，每个集群中心为一行数据

'''

src_img= cv2.imread('flower.png')
img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)

#图像二维像素转换为一维
Z = img.reshape((-1,3))
z = np.float32(Z)

## 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

k = [0,2,4,8,12,16]
#将颜色聚类，分为原始，2,4,8,12,16
for n,i in enumerate(k):
    plt.subplot(2,3,n+1)
    if n == 0:
        plt.imshow(img),plt.title('original'),plt.xticks([]),plt.yticks([])
        continue

    ret,label,center=cv2.kmeans(z,i,None,criteria,10,flags)

    center = np.uint8(center)
    res = center[label.ravel()]

    res = res.reshape(img.shape)
    plt.imshow(res),plt.title('K ={}'.format(i)),plt.xticks([]),plt.yticks([])
plt.show()
