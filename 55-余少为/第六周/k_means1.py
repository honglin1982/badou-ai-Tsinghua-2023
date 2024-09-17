import numpy as np
import cv2
import random


def k_means(src, k, attempts):
    h, w = src.shape
    data = src.flatten()

    # 初始化质心，随机生成K个下标，取到对应灰度值作为质心
    tmp_c = []  # 质心下标
    p = []  # 质心像素值
    for i in range(k):
        tmp = random.randint(0, len(data))
        if tmp not in tmp_c:
            tmp_c.append(tmp)
            p.append(data[tmp])
    print('p:', p)

    count = 0
    flg = True
    while flg and count < attempts:
        count += 1

        d = [[] for i in range(len(data))]  # 距离表
        result = np.zeros(data.shape, data.dtype)  # 类簇表

        # 计算质心与各个像素值的差
        for i, vals in enumerate(data):
            for j in range(k):
                d[i].append(abs(vals - p[j]))
            # 找出该像素值与哪个质心差值最小
            min_i = d[i].index(min(d[i]))
            # 聚集到差值最小的簇类中
            result[i] = p[min_i]

        # 重新计算质心
        _p = []  # 重新计算的质心像素值
        for i in range(k):
            vsum = 0
            vcnt = 0
            for j, vals in enumerate(result):
                if result[j] == p[i]:
                    vsum += data[j]     # 各个簇类像素值求和
                    vcnt += 1
            _p.append(int(vsum / vcnt))     # 计算均值，当作新的质心

        if _p == p:
            flg = False     # 若新的质心与上一轮的质心相同，则结束迭代
        else:   # 否则，用新的质心重新继续聚类
            p = _p
            flg = True

    print('Count:', count)
    print('p:', sorted(p))

    return result.reshape(h, w).astype(np.uint8)


# 读入灰度图
img = cv2.imread('lenna.png', 0)

# K-Means处理图片
dst = k_means(img.copy(), 4, 20)
cv2.imshow('Result', np.hstack([img, dst]))
cv2.waitKey()
