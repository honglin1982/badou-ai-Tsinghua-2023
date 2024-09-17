import numpy as np
import matplotlib.pyplot as plt


# 归一化法一：归到0~1之间
def Normalization1(x):
    return [ (float(i) - min(x)) / float(max(x) - min(x)) for i in x ]

# 归一化法二：归到-1~1之间
def Normalization2(x):
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]

# 标准化
def Z_score(x):
    s2 = sum([ (float(i)-np.mean(x)) * (i-np.mean(x)) for i in x ]) / len(x)
    return [(float(i) - np.mean(x)) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = Z_score(l)
print(n)
print(z)

plt.plot(l, cs)
plt.plot(z, cs)
plt.show()

