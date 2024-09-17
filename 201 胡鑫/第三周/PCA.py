import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

# X的每一列表示鸢尾花的特征, 每一行表示一个样本, y表示样本对应的标签
X, y = load_iris(return_X_y=True) 
# pca设置降维后主成分数目为2
pca = dp.PCA(n_components=2)
# 对X进行降维
reduced_x = pca.fit_transform(X)

'''画图'''
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
# 遍历每个样本
for i in range(len(reduced_x)):
    # 按鸢尾花的类别将降维后的数据表存在不同的列表中, 通过标签y判断
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()



