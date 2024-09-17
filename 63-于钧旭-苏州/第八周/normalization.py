import numpy as np
import matplotlib.pyplot as plt

# 归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]

# 标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]

l = np.random.normal(7, 2, 30)
l = np.round(l).astype('int')
l.sort()
l = l.tolist()
print(l)

cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)
print(cs)

n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)
plt.plot(z, cs)
plt.plot(n, cs)
plt.show()
