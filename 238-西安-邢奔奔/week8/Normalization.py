#! /usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

def Noramlization(x):
    '''

    :param x:
    :return: (x-x_min)/(x-x_max)
    '''
    return [float(i) - min(x) / (max(x) - min(x)) for i in x]

def Normalization2(x):
    '''

    :param x:
    :return:(x-x_mean)/(x_max-x_min)
    '''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


def Z_score(x):
    '''

    :param x:
    :return: (x-x_mean)/σ
    '''
    Std = np.sqrt(sum([(float(i) - np.mean(x))*(float(i) - np.mean(x)) for i in x]) / len(x))
    return [ (float(i) - np.mean(x)) / Std for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = Z_score(l)
print(n)
print(z)

plt.plot(l, cs,'r')
plt.plot(z, cs,'b')
plt.show()
