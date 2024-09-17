import numpy as np
import matplotlib.pyplot as plt

def normalization1(x):
    "归一化"
    """x = (x-x_min)/(x_max-x_min)"""
    return [float(i-min(x))/float(max(x)-min(x)) for i in x]

def normalization2(x):
    "-1~1"
    "x = (x-x_mean)/(x_max-x_min)"
    return [float(i -np.mean(x))/float(max(x)-min(x)) for i in x]

def z_score(x):
    "标准化"
    "x∗=(x−μ)/σ"
    """
    1、均方差就是标准差，标准差就是均方差

    2、方差 是各数据偏离平均值 差值的平方和 的平均数

    3、均方误差（MSE）是各数据偏离真实值 差值的平方和 的平均数
    """
    x_mean = np.mean(x)
    σ = sum([(i-x_mean)*(i-x_mean) for i in x])/len(x) #概率论中方差用来度量随机变量和其数学期望（即均值）之间的偏离程度。统计中的方差（样本方差）是各个样本数据和平均数之差的 平方和 的平均数
    return [(i-x_mean)/σ for i in x]

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)

n1 = normalization1(l)
n2 = normalization2(l)
z  = z_score(l)
fig,axs = plt.subplots(nrows = 2,ncols = 2,figsize = (12,6))
#图一
# axs[0,0].plot(l,cs)
axs[0,0].plot(n1,cs)
axs[0,0].set_title('归一化0~1')
#图二
# axs[0,1].plot(l,cs)
axs[0,1].plot(n2,cs)
axs[0,1].set_title('归一化-1~1')
#图三
# axs[1,0].plot(l,cs)
axs[1,0].plot(z,cs)
axs[1,0].set_title('标准化')
#图四
axs[1,1].plot(l,cs)
axs[1,1].set_title('原始数据')
# plt.plot(l,cs)
# plt.plot(n1,cs)
# plt.plot(n2,cs)
# plt.plot(z,cs)
plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus'] =False    #显示负号
plt.show()
