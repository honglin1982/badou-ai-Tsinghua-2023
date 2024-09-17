import numpy as np
import matplotlib.pyplot as plt
#归一化
def Normalization(x):
    Normal = []
    for i in x:
        n = (float(i)-min(x))/float(max(x)-min(x))
        Normal.append(n)
    return Normal

#标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    z = []
    x_mean=np.mean(x)
    sigma=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    for i in x:
        z.append((i-x_mean)/sigma)
    return z
 
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)
print(cs)
n=Normalization(l)
z=z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l,cs)
plt.plot(z,cs)
plt.show()
