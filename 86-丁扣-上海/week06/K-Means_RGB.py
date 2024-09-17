# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# todo 可以参考：https://blog.csdn.net/cyj972628089/article/details/122033850
pic_path = r'../file/lenna.png'
# 读取原始图像
img = cv2.imread(pic_path)

# 图像二维像素转换为一维 == todo 使用reshape将一个RGB像素点值的三个值作为一个单元
data = img.reshape((-1, 3))
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成2类
compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

# K-Means聚类 聚集成4类
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

# K-Means聚类 聚集成8类
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

# K-Means聚类 聚集成16类
compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

# K-Means聚类 聚集成64类
compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 图像转换回uint8二维类型
# 转换为uint8数据类型，将每个像素点值都赋值为当前组的中心点值
# 将center的值转换为uint8
centers2 = np.uint8(centers2)
# 使用center内的值替换原有像素点值
res = centers2[labels2.flatten()]
# #使用reshape调整替换后图像
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16', u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
