import pandas as pd

#读取CSV文件
# 参数'sep'指定了CSV文件中的分隔符为逗号，并使用正则表达式'\s*,\s*'来匹配分隔符前后的空格。
# 参数'engine'指定了使用Python解析CSV文件。
sales = pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')
X = sales['X'].values
Y = sales['Y'].values
#初始化赋值,累加次数为数据的长度
xy_sum = 0
x_sum = 0
y_sum = 0
x2_sum =0
data_len =len(X)

#计算累加值
for i in range(data_len):
    xy_sum += X[i]*Y[i]
    x_sum  += X[i]
    y_sum  += Y[i]
    x2_sum += X[i]*X[i]
#计算截距和斜率
k = ((data_len * xy_sum) - (x_sum * y_sum))/((data_len * x2_sum)-(x_sum * x_sum))
b = (y_sum - k * x_sum)/data_len
#打印斜率及截距
print(f'Coeff:{k},Intercept:{b}')