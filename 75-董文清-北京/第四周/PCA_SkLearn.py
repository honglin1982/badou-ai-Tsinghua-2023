import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris

x, y = load_iris(return_X_y=True) #x表示属性 y表示数据标签
pca = dp.PCA(n_components=2)      #加载PCA算法 设置降维为2
pca.fit(x)                        #训练
reducedX = pca.fit_transform(x)   #获取降维后的数据
print(pca.explained_variance_ratio_) #贡献率

#plt 展示
redX, redY = [], []
greenX, greenY = [], []
blueX, blueY = [], []

for i in range(len(reducedX)):

    if(y[i] == 0):
        redX.append(reducedX[i][0])
        redY.append(reducedX[i][1])
    elif(y[i] == 1):
        greenX.append(reducedX[i][0])
        greenY.append(reducedX[i][1])
    else:
        blueX.append(reducedX[i][0])
        blueY.append(reducedX[i][1])

plt.scatter(redX, redY, c='r', marker='*')
plt.scatter(greenX, greenY, c='g', marker='D')
plt.scatter(blueX, blueY, c='b', marker='X')
plt.show()

