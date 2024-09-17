from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import numpy as np

'''
linkage(y, method='single', metric='euclidean', optimal_ordering=False):
参数:
    y - ndarray
        y是距离矩阵, 可以是1维压缩向量(距离向量), 也可以是2维观测向量(坐标矩阵).
        若y是1维压缩向量, 则y必须是n个初始观测值的组合, n是坐标矩阵中成对的观测值.
        (聚类的数据, 可以是已经算好的坐标点间n个距离值, 也可以是n个d维坐标值。)
    method - str 
        表示度量类与类之间距离的模式:
        single: 最近邻 (默认值)
        average: 平均距离
        complete: 最远邻
        weighted: 重心
        ward: 对数平均
    metric - str or function 
        表示计算点与点之间距离的方式:
        euclidean: 欧式距离 (默认值)
        ... 等
    optimal_ordering - bool 
        默认值为False, 如果True, 计算结果的可视化会更直观, 但算法会变慢. 
        在数据量大的情况下, 这个参数最好设置为False.
返回值:
    Z - ndarray
        层次聚类结果编码后的矩阵, 记录了层次聚类的层次信息.
'''

'''
fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
参数:
    Z - ndarray
        linkage函数所返回的编码矩阵.
    t - scalar
        与参数criterion相关:
        对于 ‘inconsistent’(默认值)、‘distance’或‘monocrit’表示归并的阈值；
        对于‘maxclust’或‘maxclust_monocrit’表示cluster数量的最大值。
    criterion - str
        聚类的标准。
返回值:
    fcluster - ndarray
        返回输入每个坐标点所处的类编号。
'''
'''
层次聚类原理(步骤):
1. 将每个点都看成一个类, 计算两两之间的最小距离(两两计算, 选出值最小的一次计算的两个类)
2. 将距离最小的两个类合成一个新类
3. 重新计算两两之间的最小距离(只需要计算新类与其他类的距离了)
4. 重复2, 3, 直到所有类合成一类

特点：
• 凝聚的层次聚类并没有类似K均值的全局目标函数，没有局部极小问题或是很难选择初始点的问题。
• 合并的操作往往是最终的，一旦合并两个簇之后就不会撤销。
• 当然其计算存储的代价是昂贵的。
'''
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# 类与类之间的距离的度量模式为对数平均
Z = linkage(X, 'ward')
print(Z)
plt.figure(figsize=(5, 3))
dn = dendrogram(Z)


# 当聚类标准为distance时, 第二个参数t表示绝对的差值, 小于这个差值, 两个数据会被合, 
# 大于这个插值, 两个数据会被分开
# f表示的是每个点所属类的编号
f = fcluster(Z, 4, 'distance')
print(f)

# clusters的值各个类中的各个元素, 元素通过源数据中的索引体现
clusters = [np.where(i == f)[0].tolist() for i in range(1, f.max() + 1)]
print(clusters)
plt.figure(num='2')
colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # 为每个簇指定一种颜色
for i, cluster in enumerate(clusters):
    for j in cluster:
        plt.scatter(X[j][0], X[j][1], color=colors[i])
plt.show()