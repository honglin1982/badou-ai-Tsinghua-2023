import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rc('axes', unicode_minus=False)

def kMeans(src, k = 4):

    img = cv2.imread(src, 0) #读取为灰度图像
    height, width = img.shape
    data = np.float32(img.reshape(height * width, 1))
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria,10, flags)
    dst = labels.reshape((height, width))
    arr = [img, dst]
    titles = [u'原始图像', u'聚类图像 K=%d'%k]

    for i in range(len(arr)):
        plt.subplot(1, 2, i + 1)
        plt.imshow(arr[i], 'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i])
    plt.show()

def kMeansRGB(src):

    img = cv2.imread(src)
    data = np.float32(img.reshape(-1, 3))
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    arr = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    titles = [u'原始图像']
    k = 1

    for i in range(7):
        k *= 2
        compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
        centers = np.uint8(centers)
        dst = centers[labels.flatten()].reshape(img.shape)
        arr.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        titles.append(u'聚类图像 K=%d'%k)

    for i in range(len(arr)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(arr[i], 'autumn')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def kMeansNormal():

    data = np.random.randn(20, 2)
    clf = KMeans(n_clusters=3)
    dst = clf.fit_predict(data)
    x, y = [n[0] for n in data], [n[1] for n in data]
    plt.scatter(x, y, c=dst, marker='*', label=['A', 'B', 'C'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    picPath = 'lenna.png'
    #kMeans(picPath, 4)
    #kMeansRGB(picPath)
    kMeansNormal()
