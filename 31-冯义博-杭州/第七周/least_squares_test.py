import pandas as pd

# sep指定分隔符 s*指代换页，换行，回车等
datas = pd.read_csv("train_data.csv", sep='\s*,\s*', engine='python')
x = datas["X"]
y = datas["Y"]
num = len(datas)
sum_list = [0, 0, 0, 0]


# 套公式
for i in range(len(datas)):
    # x * y sum
    sum_list[0] = sum_list[0] + x[i] * y[i]
    # x sum
    sum_list[1] = sum_list[1] + x[i]
    # y sum
    sum_list[2] = sum_list[2] + y[i]
    # x 平方和
    sum_list[3] = sum_list[3] + x[i] * x[i]

k = (num * sum_list[0] - sum_list[1] * sum_list[2]) / (num * sum_list[3] - sum_list[1] * sum_list[1])
b = (sum_list[2] - k * sum_list[1]) / num
print(f"k={k}, b={b}")