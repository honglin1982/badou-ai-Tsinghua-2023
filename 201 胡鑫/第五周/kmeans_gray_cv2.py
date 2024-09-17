import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
kmeans(data, K, bestLabels, criteria, attempts, flags: int, centers=...):
参数: 
    data - ndarray
        表示聚类数据，最好是np.flloat32类型的N维点集
    K - int
        表示聚类类簇数
    bestLabels - ndarray
        表示期望输出的整数数组, 用于储存每个样本的聚类标签索引
    criteria - tuple
        表示迭代停止的模式格式为(type, max_iter, epsilon)
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempt - int
        表示重复kmeans算法的次数, 算法返回产生的最佳结果的标签
    flags - int
        表示初始质心的选择, 两种方法cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers - ndarray
        表示集群中心的输出矩阵, 每个集群中心为一行数据(每个质心点的像素值)
'''

# 以灰度图的方式读取指定图片
img = cv2.imread('../lenna.png', 0)
# 整理图片数据
data = img.reshape(img.shape[0]*img.shape[1], 1)
data = np.float32(data)
# 设置停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置随机内点选取模式
flags = cv2.KMEANS_RANDOM_CENTERS

# kmeans聚类
'''
其中, labels为聚类后每个点所属于的某个类的标签(0, 1)(在灰度图中可直接用此作为像素坐标), 
centers是每个类的质心的像素值
'''
compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags=flags)
# print(centers)
# exit()
# 由于这里是灰度图, 所以我们可以直接使用labels来表示聚类后的图像
dst = labels.reshape(img.shape)

'''
绘图
'''
plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文显示问题
titles = [u'原始图像', u'kmeans聚类图像k=2']
imgs = [img, dst]

for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
plt.show()
