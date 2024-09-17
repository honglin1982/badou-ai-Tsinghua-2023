# K-means代码思路：
# 获取一张图片的灰度图并将其二维数组转化为一维
# 先把cv2函数写出来compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)
# 然后挨个写每个参数定义

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 用cv2直接获取灰度图
img = cv2.imread('lenna.png', 0)
print(img.shape)

# 将二维的长、宽数据转为一维data，并将data转为转换为32位浮点数
h, w = img.shape[:]
data = img.reshape(h*w, 1)
data = np.float32(data)

# 挨个定义Kmeans的参数，第一个是K
K = 4

# 终止条件，eps是迭代精度误差设置为1.0，max_iter是迭代次数设置为10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 初始中心的选择，有两种方式：cv2.KMEANS_PP_CENTERS(相差最大)和cv2.KMEANS_RANDOM_CENTERS(随机生成)
flags = cv2.KMEANS_RANDOM_CENTERS

# cv2的Kmeans函数表达式
compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)

# 上述Kmeans的表达式返回的第二个值labels，以原图的形状重构
dst = labels.reshape(img.shape[0], img.shape[1])

# 以下是使用plt进行图形结果处理
# 图形字体显示中文字符
plt.rcParams['font.sans-serif'] = ['SimHei']
# 题目和图像的列表定义
titles = [u'原始图像', u'目标图像']
images = [img, dst]
# 展示图形结果
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')  # 创建一个1行2列的子图，并定位到第i+1个位置；在当前子图中显示灰度图片。
    plt.title(titles[i])  # 设置当前子图的标题为titles[i]
    plt.xticks([]), plt.yticks([])   # 移除当前子图的x轴和y轴刻度
plt.show()
