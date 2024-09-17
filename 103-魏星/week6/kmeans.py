import math
import numpy as np
import random
import matplotlib.pyplot as plt

'''
1、随机生成200个二维空间平面点
2、随机选取K个点作为初始中心
3、执行k-means聚类方法，生成K类

K值的确定？先验法、手肘法(随K值的增加，类内距离变小，曲线挂点对应的K值可作为目标K)
'''

def distiance(arr1,arr2):
    return np.linalg.norm(arr1[0]-arr2[0])+np.linalg.norm(arr1[1]-arr2[1])+np.linalg.norm(arr1[2]-arr2[2])

def kmeans(K, arrs):
    # 中心点
    center_point = [[] for i in range(0, K)]
    for i in range(0, K):
        center_point[i] = arrs[random.randint(1, 201)]
    K1=[]
    K2=[]
    K3=[]

    intergeneration = 0
    new_point = [[0 for j in range(1, 3)] for i in range(0, K)]
    while(distiance(np.array(center_point), np.array(new_point)) > 0.01):
        for i in range(0, K):
            new_point[i] = center_point[i]
        K1.clear()
        K2.clear()
        K3.clear()
        intergeneration = intergeneration+1
        for i in range(arrs.shape[0]):
            sqrt1 = math.sqrt(
                math.pow(arrs[i][0] - center_point[0][0], 2) + math.pow(arrs[i][1] - center_point[0][1], 2))
            sqrt2 = math.sqrt(
                math.pow(arrs[i][0] - center_point[1][0], 2) + math.pow(arrs[i][1] - center_point[1][1], 2))
            sqrt3 = math.sqrt(
                math.pow(arrs[i][0] - center_point[2][0], 2) + math.pow(arrs[i][1] - center_point[2][1], 2))
            if sqrt1 == min(sqrt1, sqrt2, sqrt3):
                K1.append(arrs[i])
            elif sqrt2 == min(sqrt1, sqrt2, sqrt3):
                K2.append(arrs[i])
            else:
                K3.append(arrs[i])

        center_point[0] = np.mean(K1, axis=0)
        center_point[1] = np.mean(K2, axis=0)
        center_point[2] = np.mean(K3, axis=0)

        print("第{}次循环",intergeneration)
    return new_point,np.array(K1),np.array(K2),np.array(K3)


def do_kmeans():
    K=3
    a = [[random.uniform(-10, 10) for j in range(1, 3)] for i in range(1, 201)]
    a_arr = np.array(a)
    # print(a_arr)

    center_point,k1,k2,k3 = kmeans(K, a_arr)
    print(center_point)
    # print(k1.shape)
    # print(k2.shape)
    # print(k3.shape)

    color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
    plt.figure()
    plt.scatter(center_point[0][0], center_point[0][1], c=color[1], marker='o',s=150)
    plt.scatter(center_point[1][0], center_point[1][1], c=color[2], marker='o',s=150)
    plt.scatter(center_point[2][0], center_point[2][1], c=color[3], marker='o',s=150)
    plt.scatter(k1[:,0], k1[:,1], c=color[1], marker='o', label='k1')
    plt.scatter(k2[:,0], k2[:,1], c=color[2], marker='o', label='k2')
    plt.scatter(k3[:,0], k3[:,1], c=color[3], marker='o', label='k3')
    plt.legend(loc=2)
    plt.show()

if __name__ == '__main__':
     do_kmeans()

