from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

'''
KMeans原理(步骤):
    1. 确定k值, 将数据集聚集成K个簇  
    2. 在所有点中随机选取k个质心点
    3. 计算每个点到每个质心的距离, 并将每个点划分到最近质心的簇
    4. 当每个质心都聚集了一些点后, 重新定义算法选出新的质心
       (对于每个簇, 计算其均值, 即得到新的k个质心点)
    5. 迭代执行第三第四步, 直到满足迭代终止条件为止(聚类结果不再变化)
'''



'''
数据集X:
    篮球运动员比赛数据
    第一列表示球员每分钟助攻数: assists_per_minute
    第二列表示球员每分钟得分数: points_per_minute
'''

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
     [0.1956, 0.4280]]

# 得到KMeans聚类(k=3)的训练模型
clf = KMeans(n_clusters=3)
# 训练数据集并得到聚类结果
y_pred = clf.fit_predict(X)
print(y_pred)

'''
绘图
'''
x = [n[0] for n in X]   
y = [n[1] for n in X]

plt.title("player data")
plt.scatter(x, y, c=y_pred, marker="x")
plt.title('kmeans basketball player data')
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')
plt.show()