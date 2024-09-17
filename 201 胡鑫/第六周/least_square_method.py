import numpy as np
import pandas as pd

data = pd.read_csv('train_data.csv')
X = data['X'].values
Y = data['Y'].values

# 初始化赋值
s1 = s2 = s3 = s4 = 0
# 数据量
n = len(X)

# 根据公式赋值
for i in range(n):
    # sum(xn*yn)
    s1 += X[i] * Y[i]
    # sum(xn)
    s2 += X[i]
    # sum(yn)
    s3 += Y[i]
    # sum(xn**2)
    s4 += X[i] * X[i]

# 套公式计算k，b
k = (n * s1 - s2 * s3) / (n * s4 - s2 ** 2)
b = (s3 - k*s2) / n

print(f'fit :  y = {k}x + {b}')