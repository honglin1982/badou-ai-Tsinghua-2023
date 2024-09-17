from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

'''
对于数据集内的任意一点, 有一个epsilon邻域, 当这个邻域中的点的个数大于等于一个阈值时, 
将此点epsilon邻域内的所有点聚成一类, 否则将此点标记为杂音
注: 某一点在自己epsilon邻域内不满足条件, 但是可能满足其他的点的条件, 当满足时, 会和
其他满足条件的点聚成一类, 也就不再是杂音点了
'''

iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)

# 每个点周围的eps邻域内的点大于等于9个时聚为一类
dbscan = DBSCAN(eps=0.4, min_samples=9)
labels = dbscan.fit_predict(X)

print(labels)

'''
绘图
'''
x0 = X[labels == 0]
x1 = X[labels == 1]
x2 = X[labels == 2]
x3 = X[labels == -1]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2') 
plt.scatter(x3[:, 0], x3[:, 1], c="magenta", marker='+', label='label3')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  
