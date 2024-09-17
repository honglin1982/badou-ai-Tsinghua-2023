import matplotlib.pyplot as plt
import numpy as np
import random

'''
[xmin,xmax]映射到[ymin,ymax]
线性归一化的一般规范函数是：y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin 
'''
## 映射到0-1
def normalization01(arr):
    max = np.max(arr)
    min = np.min(arr)
    return [ (arr[i] - min)/(max - min) for i in arr]

## 映射到-1到1
def normalization11(arr):
    max = np.max(arr)
    min = np.min(arr)
    return [ (2*(arr[i] - min) / (max - min)) -1 for i in arr]

'''
标准差标准化
x = (x-mean)/std
'''
def z_score_normalization(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return [(arr[i]-mean)/std for i in arr]


a = [random.randint(-16, 12) for i in range(1, 51)]
a_arr = np.array(a)
x = [ i for i in range(1, 51)]
normalTo01 = normalization01(a_arr)
normalTo11 = normalization11(a_arr)
z_score = z_score_normalization(a_arr)
print(x)
print(normalTo01)
print(normalTo11)
print(z_score)

plt.figure()
plt.rcParams['font.sans-serif']=['SimHei'] ##可以使用中文
plt.scatter(x, a_arr, marker='o', label='原始数据')
plt.scatter(x, normalTo01, marker='o', label='归一化0-1')
plt.scatter(x, normalTo11, marker='o', label='归一化-1-1')
plt.scatter(x, z_score, marker='o', label='标准差标准化')
plt.legend(loc=2)
plt.show()


