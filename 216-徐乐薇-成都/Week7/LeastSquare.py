# 最小二乘法
import pandas as pd

sales = pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python') # 读取数据,"\s*,\s*"表示以,分割,engine='python'表示用python引擎读取
print(sales.head()) # 打印前5行数据
print(sales.shape) # 打印数据的行列数
print(sales.columns) # 打印列名

x = sales['X'].values # 获取x列的值，values表示获取值，['x']表示获取x列
y = sales['Y'].values # 获取y列的值

# 初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4 # 数据的个数,这里是4个

# 循环累加
for i in range(n):
    s1 += x[i] * y[i]
    s2 += x[i]
    s3 += y[i]
    s4 += x[i] * x[i]

# 根据求导得极值点的公式。计算k,b的值
k = (n * s1 - s2 * s3) / (n * s4 - s2 * s2)
b = (s3 - k * s2) / n
print('k=',k,'b=',b)
# y = kx + b = 1.4x + 3.5