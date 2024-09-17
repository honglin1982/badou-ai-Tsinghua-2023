# coding=utf-8  
from sklearn.cluster import KMeans
import os
import numpy as np
 
 
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]   
    ]

def my_kmeans(X, n_clusters=2):
    max_x = max([x[0] for x in X])
    max_y = max([x[1] for x in X])

    class_pt = [[0.05, 0.2], [0.2, 0.8]]
    # for i in range(n_clusters):
    #     pt_x = i+1/(n_clusters + 1) * max_x
    #     pt_y = i+1/(n_clusters + 1) * max_y

    #     class_pt.append([pt_x, pt_y])

    # print(class_pt)
    
    y_pred = np.zeros(len(X))
    threshold = 0.00001
    delta = 1
    cnt = 0
    while delta > threshold and cnt < 100:
        for ix, x in enumerate(X):
            flag = -1
            tmp_list = []
            for pt in class_pt:
                tmp_list.append((x[0]-pt[0])**2 + (x[1]-pt[1])**2)
            for i, dist in enumerate(tmp_list):
                if dist == min(tmp_list):
                    flag = i
            y_pred[ix] = flag 
        # print(y_pred)
        delta_list = []
        for i in range(n_clusters):
            tmp_list = []
        
            for ix, x in enumerate(X):
                if y_pred[ix] == i:
                    tmp_list.append(x)
            tmp_list = np.array(tmp_list)

            if len(tmp_list) == 0:
                continue
            for t in tmp_list:
                delta_list.append(abs((t[0]-class_pt[i][0]) + (t[1]-class_pt[i][1])))

            tmp_pt_x = np.mean(tmp_list[:, 0])
            tmp_pt_y = np.mean(tmp_list[:, 1])
            class_pt[i][0] = tmp_pt_x
            class_pt[i][1] = tmp_pt_y

        delta = np.mean(delta_list)
        cnt += 1
        # print(delta_list)
        print(f"delta: {delta}, cnt: {cnt}")
        print(f"class_pt: {class_pt}")
        print("-------------------------------------")
        if cnt == 3:
            return y_pred
    return y_pred

"""
第二部分：KMeans聚类
clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
"""

# clf = KMeans(n_clusters=2)
# y_pred = clf.fit_predict(X)

y_pred = my_kmeans(X, 2)
 
#输出完整Kmeans函数，包括很多省略参数
# print(y_pred)
#输出聚类预测结果
print("y_pred = ",y_pred)
 
"""
第三部分：可视化绘图
"""

import numpy as np
import matplotlib.pyplot as plt

#获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print (x)
y = [n[1] for n in X]
print (y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=y_pred, marker='x')
 
#绘制标题
plt.title("Kmeans-Basketball Data")
 
#绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
 
#设置右上角图例
plt.legend(["A","B","C"])
 
#显示图形
plt.show()
