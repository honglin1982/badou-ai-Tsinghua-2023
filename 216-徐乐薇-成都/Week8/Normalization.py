import numpy as np
import matplotlib.pyplot as plt

#归一化的两种方式
def normalization_1(x):
    '''归一化（0-1）'''
    '''x = (x - min) / (max - min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]
# def normalization_2(x):
#     '''归一化（-1-1）'''
#     '''x = (x - mean) / std'''
#     return [(x - np.mean(x)) / (np.max(x) - np.min(x)) for i in x]#均值为0，方差为1

#标准化
def z_score(x):
    '''标准化（-1-1）'''
    '''x = (x - mean) / std'''
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - np.mean(x)) / s2 for i in x] #均值为0，方差为1

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
# l1=[] #存放归一化后的数据
# # for i in l:
# #     i+=2
# #     l1.append(i)
# # print(l1)
cs=[] #统计每个数出现的次数
for i in l:
    c=l.count(i) #统计i出现的次数
    cs.append(c) #将次数添加到cs中
print(cs)
n=normalization_1(l) #归一化
z=z_score(l) #标准化
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l,cs) #绘制原始数据,plot(l,cs)表示将l和cs绘制在同一张图上
plt.plot(z,cs) #绘制标准化后的数据
plt.show()