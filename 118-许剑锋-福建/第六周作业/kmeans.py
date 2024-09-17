'''
kmeans
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def open_cv_kmeans(image, label_count):
    width, height = image.shape
    data = image.reshape((width * height, 1))
    data = np.float32(data)
    # 停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, label_count, None, criteria, 10, flags)

    # 生成最终图像
    dst = labels.reshape((width, height))
    print(np.unique(dst))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    titles = [u'原始图像', u'聚类图像']
    images = [image, dst]
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()



def generate_k(data, k):
    '''
    初始化质心
    :param data:
    :param k:
    :return:
    '''
    n = len(data)
    d = set()
    while len(d) != k:
        val = np.random.randint(0, n)
        if val not in d:
            d.add(val)
    center_idx = list(d)
    center_data = data[center_idx]
    return center_data



def kmeans_detail(image, label_count, iterator):
    width, height = image.shape[0], image.shape[1]
    data = image.reshape((width * height, 1))
    # 初始化质心
    center = generate_k(data, label_count)
    # 计算距离，划分类别
    label_class = [[] for _ in range(label_count)]
    ite = 0
    while ite < iterator:
        print('ite======:{}'.format(ite))
        label_class = [[] for _ in range(label_count)]
        for i in range(len(data)):
            dis = 256
            min_idx = -1
            for j in range(label_count):
                if abs(data[i] - center[j]) < dis:
                    dis = abs(data[i] - center[j])
                    min_idx = j
            label_class[min_idx].append(i)
        # 重新计算质心，与旧质心比较，相同停止，否则重复分类
        flag = True
        new_center = [0 for i in range(label_count)]
        for i in range(label_count):
            new_center[i] = np.mean(data[label_class[i]])
            if new_center[i] not in center:
                flag = False
        if flag:
            break
        center = new_center
        ite += 1


    print('迭代结束，共迭代：{}次'.format(ite))
    new_data = np.zeros(data.shape)
    new_data2 = np.zeros(data.shape)
    for i in range(label_count):
        if len(label_class[i]) == 0:
            continue
        for j in label_class[i]:
            new_data[j] = center[i]
            new_data2[j] = i
    new_data = new_data.reshape((width, height))
    new_data2 = new_data2.reshape((width, height))
    print('new_data:{}'.format(np.unique(new_data)))
    print('new_data2:{}'.format(np.unique(new_data2)))

    plt.rcParams['font.sans-serif'] = ['SimHei']

    titles = [u'原始图像', u'聚类图像-像素类别', u'聚类图像-0123']
    images = [image, new_data, new_data2]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    image = cv2.imread('lenna.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    open_cv_kmeans(gray, 4)
    kmeans_detail(gray, 4, 20)
