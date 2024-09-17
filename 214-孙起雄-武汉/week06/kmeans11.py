import numpy as np


def kmeans(data, k, max_iters=100):

    # 随机选择k个样本作为初始的簇中心
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iters):
        # 计算每个样本到各个簇中心的距离
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))

        # 分配样本到距离最近的簇
        labels = np.argmin(distances, axis=0)

        # 更新簇中心
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # 判断是否收敛
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


# 示例数据
data = np.array([[1, 2], [2, 1], [1, 1], [3, 2], [4, 3], [5, 4]])

# 调用K-means算法
centroids, labels = kmeans(data, k=2)

# 输出结果
print("簇中心点:")
print(centroids)
print("样本标签:")
print(labels)
