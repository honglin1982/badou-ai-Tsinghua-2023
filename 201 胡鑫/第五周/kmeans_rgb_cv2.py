import cv2
import numpy as np
from matplotlib import pyplot as plt

# 以rgb模式读取图片
img = cv2.imread('../lenna.png', 1)
# 格式化数据
data = img.reshape(img.shape[0]*img.shape[1], 3)
data = np.float32(data)
# 停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 质心选取模式
flags = cv2.KMEANS_RANDOM_CENTERS

# kmeans聚类k=2
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags=flags)
'''
此时是rgb图像, 每个点的像素值由三个值一起表示, 因此不能仅仅使用所得到的标签来表示图像
需要进行下列操作
'''
# print(labels2)
# print(labels2.flatten())
centers2 = np.uint8(centers2)
# labels.flatten()的作用为整理数据, 将labels转换成一维行向量, 方便后续操作
# 表示得出每个点属于某一类的像素值, 每个点都使用所属类的质心的像素值, 结果为每个点的像素值
res = centers2[labels2.flatten()]
dst2 = res.reshape(img.shape)
# print(dst2.shape)

'''
同理得出k为其他值时候的dst, 随便取几组
'''
# k=8
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags=flags)
centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape(img.shape)
# k=16
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags=flags)
centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape(img.shape)
# k=32
compactness, labels32, centers32 = cv2.kmeans(data, 32, None, criteria, 10, flags=flags)
centers32 = np.uint8(centers32)
res = centers32[labels32.flatten()]
dst32 = res.reshape(img.shape)
# k=128
compactness, labels128, centers128 = cv2.kmeans(data, 128, None, criteria, 10, flags=flags)
centers128 = np.uint8(centers128)
res = centers128[labels128.flatten()]
dst128 = res.reshape(img.shape)


# 用列表储存所有图片
imgs = [img, dst2, dst8, dst16, dst32, dst128]
# bgr ------> rgb
for i in range(6):
    imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
'''
绘图
'''
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'k=2', u'k=8', u'k=16', u'k=32', u'k=128']
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(imgs[i], cmap='gray'), plt.title(titles[i])
plt.show()