import pandas as pd
import matplotlib.pyplot as plt


# 指定python引擎读取csv文件
samples = pd.read_csv("image/train_data.csv", sep=",", engine="python")

x = samples["X"].values
y = samples["Y"].values
print(x, y)

# 计算截距与斜率
n = len(x)  # 数据量
x_y = sum(x * y)  # x*y的累加和
x_ = sum(x)  # x的累加和
y_ = sum(y)  # y的累加和
x_2 = sum(x * x)   # x^2 的累加和
print(x_y, x_, y_, x_2)

k = (x_*y_ - n*x_y) / (x_*x_ - x_2*n)
b = (y_ - k*x_) / n
print(f"斜率：{k}, 截距：{b}\n数学模型：y={k}x+{b}")

# 创建散点图
plt.scatter(x, y, c="green")
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")

# 画出拟合直线
x_line = [min(x)-1, max(x)+1]
y_line = [k*x_line[0]+b, k*x_line[1]+b]
plt.plot(x_line, y_line, color="red")

plt.show()
