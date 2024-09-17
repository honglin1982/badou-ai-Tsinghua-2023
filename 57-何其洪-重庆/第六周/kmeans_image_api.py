# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png', 0)
    # 转换为3列
    data = img.reshape((int((img.shape[0] * img.shape[1]) / 2), 2))
    data = np.float32(data)
    """
    参数: 
        data:  需要分类数据，最好是np.float32的数据，每个特征放一列。
        K:  聚类个数 
        bestLabels：预设的分类标签或者None
        criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为(type, max_iter, epsilon) 其中，type有如下模式：
            cv2.TERM_CRITERIA_EPS ：精确度（误差）满足epsilon，则停止。
            cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter，则停止。
            cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER：两者结合，满足任意一个结束
        attempts：重复试验kmeans算法次数，将会返回最好的一次结果
        flags：初始中心选择，可选以下两种：
            cv2.KMEANS_PP_CENTERS：使用kmeans++算法的中心初始化算法，即初始中心的选择使眼色相差最大
            cv2.KMEANS_RANDOM_CENTERS：每次随机选择初始中心
    返回值:
        compactness：紧密度，返回每个点到相应重心的距离的平方和
        labels：结果标记，每个成员被标记为分组的序号，如 0,1,2,3,4...等
        centers：由聚类的中心组成的数组
    """
    # 定义终止条件 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    result = centers[labels.flatten()]
    result = result.reshape(img.shape)
    cv2.imshow("1", result)
    cv2.waitKey(0)
