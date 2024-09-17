import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png')
data = img.reshape((-1, 3))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_PP_CENTERS + cv2.KMEANS_USE_INITIAL_LABELS  # cv2.KMEANS_RANDOM_CENTERS
'''
cv2.KMEANS_PP_CENTERS：使用K-Means++算法进行聚类中心的初始化。K-Means++算法会尽量选择距离较远的点作为初始聚类中心，以提高聚类效果。
cv2.KMEANS_PP_CENTERS + cv2.KMEANS_USE_INITIAL_LABELS：结合K-Means++算法和使用初始标签的方法进行聚类中心的初始化。初始标签可以通过之前的聚类结果作为输入，以加快收敛速度。
cv2.KMEANS_USE_INITIAL_LABELS：使用指定的初始标签进行聚类中心的初始化。初始标签可以通过之前的聚类结果作为输入，以加快收敛速度。
'''

compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 图像转换回uint8二维类型(整数)
centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]  # 通过索引操作，根据每个样本的聚类标签获取对应的聚类中心点
# flatten()是将数组进行展平成一维，也就是展平为一行
dst4 = res.reshape(img.shape)

dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

plt.subplot(1, 1, 1), plt.imshow(dst4, 'gray'), plt.xticks([]), plt.yticks([])
# 灰度颜色映射只适用于灰度图像，对于彩色图像，指定灰度颜色映射不会改变图像本身的颜色
plt.show()
