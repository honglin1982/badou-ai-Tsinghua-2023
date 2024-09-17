# 最小二乘法
import numpy as np

X = [1,2,3,4]
Y = [6,5,7,10]

s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4

for i in range(n):
    s1 += X[i] * Y[i]
    s2 += X[i]
    s3 += Y[i]
    s4 += X[i]*X[i]

# 计算斜率和截距，公式见ppt
k = (s2 * s3 - n * s1) / (s2 * s2 - s4 * n)
b = (s3 - k * 2) / n

print("斜率: {} 截距: {}".format(k, b)) # 斜率: 1.4 截距: 6.3

