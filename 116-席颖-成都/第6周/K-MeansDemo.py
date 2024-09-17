# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import cv2

def KMeans_athlete():
    #数据集
    data = np.random.randn(30,2)
    print(data)
    clf  = KMeans(n_clusters=3)#表示类簇数为3，聚成3类数据，clf即赋值为KMeans
    y_pred = clf.fit_predict(data)#载入数据集X，并且将聚类的结果赋值给y_pred
    #绘制图形
    x = [n[0] for n in data]
    y = [n[1] for n in data]
    plt.scatter(x,y,c=y_pred,marker='*')
    plt.title("KMeans-random data")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(['A','B','C'])
    plt.show()

def K_Means_RGB(src):
    #读取原始图像
    img = cv2.imread(src)
    print(img.shape)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 图像二维像素转换为一维,3列
    data = img.reshape(-1,3)
    data = np.float32(data)
    #print(data)
    #停止条件
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    arr      = [cv2.cvtColor(img,cv2.COLOR_BGR2RGB)]
    titles   = [u'原图']
    k = 1

    for i in range(7):
        k = k *2
        compactness ,labels,centers = cv2.kmeans(data,k,None,criteria,10,flags)
        # 图像转换回uint8二维类型
        centers = np.uint8(centers)
        dst     = centers[labels.flatten()].reshape(img.shape)
        arr.append(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
        titles.append(u'聚类 k=%d'%k)

    for i in range(len(arr)):
        plt.subplot(2,4,i+1)
        plt.imshow(arr[i],'autumn')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def kMeans(src, k = 4):
    '''
    在OpenCV中，Kmeans()函数原型如下所示：
    retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
        data表示聚类数据，最好是np.flloat32类型的N维点集
        K表示聚类类簇数
        bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
        criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
            其中，type有如下模式：
             —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
             —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
             —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
        attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
        flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
        centers表示集群中心的输出矩阵，每个集群中心为一行数据
    '''
    img = cv2.imread(src, 0) #读取为灰度图像
    height, width = img.shape
    data = np.float32(img.reshape(height * width, 1))
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria,10, flags)
    dst = labels.reshape((height, width))
    arr = [img, dst]
    titles = [u'原始图像', u'聚类图像 k=%d'%k]

    for i in range(len(arr)):
        plt.subplot(1, 2, i + 1)
        plt.imshow(arr[i], 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
    plt.show()

if __name__ == '__main__':
    #KMeans_athlete()
    #K_Means_RGB('lenna.png')
    kMeans('lenna.png',4)


