import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets._base import load_iris

x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签
print(x.shape,y.shape)
pca=PCA(n_components=3) #加载pca算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x) #对原始数据进行降维，保存在reduced_x中
print(reduced_x.shape)
red_x,red_y,red_z=[],[],[]
blue_x,blue_y,blue_z=[],[],[]
green_x,green_y,green_z=[],[],[]
for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
        red_z.append(reduced_x[i][2])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
        blue_z.append(reduced_x[i][2])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
        green_z.append(reduced_x[i][2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(red_x,red_y,red_z,c='r',marker='x')
ax.scatter(blue_x,blue_y,blue_z,c='b',marker='D')
ax.scatter(green_x,green_y,green_z,c='g',marker='.')
plt.show()
