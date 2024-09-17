from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

'''
实现k-means
'''
import random


def Kmeans(data, k, iter_num=10):
    data = np.array(data)
    num = data.shape[0]
    clusters = defaultdict(list)
    assert num >= k

    # 初始化如下
    zhixin_list = np.array(random.sample(list(data), k))      # 记录质心

    distance_matrix = np.zeros((num, k))
    for i in range(num):
        for j in range(k):
            distance_matrix[i, j] = np.sqrt(np.sum((data[i] - zhixin_list[j]) * (data[i] - zhixin_list[j])))
    cluster_index_old = np.zeros((num, 1))
    cluster_index_new = np.argmin(distance_matrix, axis=-1)  # 记录新的每个样本所属的类别
    for i in range(num):
        clusters[cluster_index_new[i]].append(list(data[i]))

    while iter_num > 0 and (cluster_index_old != cluster_index_new).any():
        cluster_index_old = cluster_index_new
        # 选择新质心
        for i in range(len(clusters)):
            zhixin_list[i] = np.average(np.array(clusters[i]), axis=0)
        # 重新分类
        for i in range(num):
            for j in range(k):
                distance_matrix[i, j] = np.sqrt(np.sum((data[i] - zhixin_list[j]) * (data[i] - zhixin_list[j])))
        cluster_index_new = np.argmin(distance_matrix, axis=-1)
        clusters = defaultdict(list)
        for i in range(num):
            clusters[cluster_index_new[i]].append(list(data[i]))
        iter_num = iter_num - 1
    if iter_num == 0:
        print("迭代超过%d次" % iter_num)

    return clusters, zhixin_list, cluster_index_new


data = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.9747, 0.4974],
     [0.7983, 0.1772],
     [0.4276, 0.5703],
     [3.1671, 3.9835],
     [3.5306, 3.5276],
     [3.1061, 3.7523],
     [3.7446, 3.4007],
     [3.9670, 3.2770],
     [6.2485, 6.8313],
     [6.1227, 6.9909],
     [6.5240, 6.5668],
     [6.7461, 6.3113],
     [6.6315, 6.1788],
     [8.0494, 8.7590],
     [8.1107, 8.9799],
     [8.4121, 8.4735],
     [8.5007, 8.7318],
     [8.0567, 8.0326],
     # [10.1956, 10.4280]
    ]
clusters, zhixin_list, cluster_index = Kmeans(data, 4)
print(clusters)
for i in range(len(clusters)):
    print("第%d类" % (i + 1))
    print(clusters[i])
print()
print(zhixin_list)
print(cluster_index)

plt.scatter([e[0] for e in data], [e[1] for e in data], c=cluster_index, marker='x')
plt.show()