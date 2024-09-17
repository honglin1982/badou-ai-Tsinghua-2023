# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引,各个数据点的最终分类标签（索引）
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''


class CV2KMeans(object):
    """ K-means 聚类 """

    def __init__(self, pic_path=None, K=None, bestLabels=None, criteria=None, number=10, flags=None):
        self.pic_path = pic_path
        self.K = K
        self.img_gray = self.get_img_gray()
        self.data = self.latitude()
        # 停止条件 (type,max_iter,epsilon)
        self.criteria = criteria
        self.bestLabels = bestLabels
        self.number = number
        self.flags = flags

    def get_img_gray(self) -> np.ndarray:
        return cv2.imread(self.pic_path, 0)

    def latitude(self):
        # 获取图像高度、宽度
        high, weight = self.img_gray.shape
        # 图像二维像素转换为一维
        data = self.img_gray.reshape((high * weight, 1))
        return np.float32(data)

    def k_means(self):
        # K-Means聚类 聚集成4类
        compactness, labels, centers = cv2.kmeans(self.data, self.K, self.bestLabels, self.criteria, self.number, self.flags)
        return compactness, labels, centers

    def ret_result(self):
        compactness, labels, centers = self.k_means()  # labels： 各个数据点的最终分类标签（索引）
        # 生成最终图像
        dst = labels.reshape((self.img_gray.shape[0], self.img_gray.shape[1]))
        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 显示图像
        titles = [u'原始图像', u'聚类图像']
        images = [self.img_gray, dst]
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == '__main__':
    pic_path = r'../file/lenna.png'
    # 停止条件 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    ckm_obj = CV2KMeans(pic_path, K=4, bestLabels=None, criteria=criteria, number=10, flags=flags)
    ckm_obj.ret_result()
    pass
