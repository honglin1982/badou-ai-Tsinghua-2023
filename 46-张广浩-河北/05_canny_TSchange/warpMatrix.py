#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 8:55
# @Author  : zgh
# @FileName: warpMatrix.py
# @Software: PyCharm
import numpy as np


def WarpPerspectiveMatrix(input, target):
	"""
	:param input:[(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
	:param target: [(X'0,Y'0),(X'1,Y'1),(X'2,Y'2),(X'3,Y'3)]
	:return:
	"""
	# 保证输入数据是4个点以上，我们需要确定8个参数，4个点8个值8
	assert input.shape[0] >= 4 and target.shape[0] >= 4
	row = input.shape[0]
	# 构建A矩阵，8行8列
	A = np.zeros((row * 2, 8))
	# 构建B矩阵，8行1列
	B = np.zeros((row * 2, 1))
	# 遍历每个坐标
	for i in range(row):
		# 获取(x,y)  (X',Y')
		input_i = input[i]
		target_i = target[i]
		# 对应位置赋值
		A[2 * i, :] = [input_i[0], input_i[1], 1, 0, 0, 0, -input_i[0] * target_i[0], -input_i[1] * target_i[0]]
		B[2 * i, :] = target_i[0]
		A[2 * i + 1, :] = [0, 0, 0, input_i[0], input_i[1], 1, -input_i[0] * target_i[1], -input_i[1] * target_i[1]]
		B[2 * i+1, :] = target_i[1]
	# A转化矩阵(8,8)
	A = np.mat(A)
	# A的逆乘以B (8,8) * (8,1) = (8,1)
	warpmatrix = A.I * B
	# 行列变化
	warpmatrix = warpmatrix.reshape(1, 8)
	# 最后一个位置加上1，也就是赋予a33 = 1
	warpmatrix = np.insert(warpmatrix,warpmatrix.shape[1], values=1)
	# reshape成3,3矩阵
	warpmatrix = np.reshape(warpmatrix, (3, 3))
	return warpmatrix


if __name__ == '__main__':
	print('warpMatrix')
	src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
	src = np.array(src)

	dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
	dst = np.array(dst)

	warpMatrix = WarpPerspectiveMatrix(src, dst)
	print(warpMatrix)
