import numpy as np
import matplotlib.pyplot as plt


# 1、归一化（0~1）: x_=(x−x_min)/(x_max−x_min)
def Normalization1(x):
    x = [float(i - min(x)) / float(max(x) - min(x)) for i in x]
    return x

# 2、归一化（-1~1）: x_=(x−x_mean)/(x_max−x_min)
def Normalization2(x):
    x = [(float(i) - np.mean(x)) / float((max(x) - min(x))) for i in x]
    return x


# 3、标准化: y=(x−μ)/σ
def z_score(x):
    # • 经过处理后的数据均值为0，标准差为1（正态分布）
    # • 其中μ是样本的均值， σ是样本的标准差
    u = np.mean(x)  # 平均值
    var = sum([(i - u) ** 2 for i in x]) / len(x)  # 方差
    o = np.sqrt(var)  # 标准差
    print(f"平均值={u}  方差={var}  标准差={o}")
    x = [(i - u) / o for i in x]
    return x


# 原始数据
data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
        11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
# 计算列表中每个元素重复的个数
data_cnt = [data.count(i) for i in data]
print("元素个数", data_cnt)

n1 = Normalization1(data)
n2 = Normalization2(data)
z = z_score(data)
print("n1:", n1)
print("n2:", n2)
print("z:", z)

'''
蓝线为原始数据，橙线为z
'''
plt.plot(data, data_cnt)
# plt.plot(n1, data_cnt)
# plt.plot(n2, data_cnt)
plt.plot(z, data_cnt)
plt.show()
