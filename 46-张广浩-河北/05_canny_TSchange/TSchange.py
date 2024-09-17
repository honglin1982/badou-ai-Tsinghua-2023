#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 10:00
# @Author  : zgh
# @FileName: TSchange.py
# @Software: PyCharm

import cv2 as cv
import numpy as np


# 手动实现
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
		B[2 * i + 1, :] = target_i[1]
	# A转化矩阵(8,8)
	A = np.mat(A)
	# A的逆乘以B (8,8) * (8,1) = (8,1)
	warpmatrix = A.I * B
	# 行列变化
	warpmatrix = warpmatrix.reshape(1, 8)
	# 最后一个位置加上1，也就是赋予a33 = 1
	warpmatrix = np.insert(warpmatrix, warpmatrix.shape[1], values=1)
	# reshape成3,3矩阵
	warpmatrix = np.reshape(warpmatrix, (3, 3))
	return warpmatrix


def get_warpmatrix(src, dst, type=0):
	if type == 0:
		warpmatrix = WarpPerspectiveMatrix(src, dst)
		print("手动实现获取warpmatrix函数")
	else:
		warpmatrix = cv.getPerspectiveTransform(src=src, dst=dst)
		print("调用接口实现获取warpmatrix函数")
	print(warpmatrix.shape)
	print(warpmatrix)
	return warpmatrix


if __name__ == '__main__':
	img = cv.imread("img.jpg")
	img_copy = img.copy()
	src = [[503, 36], [877, 172], [427, 1262], [60, 1103]]
	dst = [[0, 0], [370, 0], [370, 1100], [0, 1100]]
	src = np.float32(src)
	dst = np.float32(dst)
	# 调用接口
	warpmatrix = get_warpmatrix(src, dst, type=1)

	result = cv.warpPerspective(img_copy, warpmatrix, (370, 1100))
	cv.imshow("img", img)
	cv.imshow("result", result)
	cv.imwrite("result.jpg ", result)
	cv.waitKey(0)
