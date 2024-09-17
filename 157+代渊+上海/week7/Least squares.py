import pandas as pd

sales=pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')  #读取CSV
X=sales['X'].values    #存csv的第一列
Y=sales['Y'].values    #存csv的第二列

#初始化赋值
sum_XY = 0
sum_X = 0
sum_Y = 0
sum_X2 = 0
n = 4       ####你需要根据的数据量进行修改

#循环累加
for i in range(n):
    sum_XY = sum_XY + X[i]*Y[i]     #X*Y，求和
    sum_X = sum_X + X[i]          #X的和
    sum_Y = sum_Y + Y[i]          #Y的和
    sum_X2 = sum_X2 + X[i]*X[i]     #X**2，求和

#计算斜率和截距
k = (sum_X*sum_Y-n*sum_XY)/(sum_X*sum_X-sum_X2*n)
b = (sum_Y - k*sum_X)/n
print("Coeff: {} Intercept: {}".format(k, b))
#y=1.4x+3.5
