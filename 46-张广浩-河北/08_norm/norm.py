# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 8:58
# @Author  : zgh
# @FileName: norm.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt


# 均匀归一化,数据归到0 - 1之间
def normalization_min(x):
	res_x = [(float(i) - min(x)) / float((max(x) - min(x))) for i in x]
	return res_x


# 均值归一化,数据归到-1 - 1之间
def normalization_mean(x):
	res_x = [(float(i) - np.mean(x)) / float((max(x) - min(x))) for i in x]
	return res_x


# 标准化
def z_score(x):
	x_mean = np.mean(x)
	sum_x = sum([(i - x_mean) ** 2 for i in x]) / (len(x))
	res_x = [(i - x_mean) / np.sqrt(sum_x) for i in x]
	print("均值", np.mean(res_x))
	print("方差", np.var(res_x))
	return res_x


if __name__ == '__main__':
	# x = np.random.randint(-2.5, 2.5, 20)
	x = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
	     11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

	cs = []
	for i in x:
		c = x.count(i)
		cs.append(c)
	normalization_min_x = normalization_min(x)
	normalization_mean_x = normalization_mean(x)
	z_score_x = z_score(x)
	res_min_x = [round(i, 2) for i in normalization_min_x]
	res_mean_x = [round(i, 2) for i in normalization_mean_x]
	res_z_x = [round(i, 2) for i in z_score_x]
	print(x)
	print(res_min_x)
	print(res_mean_x)
	print(res_z_x)

	plt.plot(x, cs, label='source_x')
	plt.plot(res_min_x, cs, label='normalization_min_x')
	plt.plot(res_mean_x, cs, label='normalization_mean_x')
	plt.plot(res_z_x, cs, label='standardization_x')

	plt.legend(loc='upper right')
	plt.show()
