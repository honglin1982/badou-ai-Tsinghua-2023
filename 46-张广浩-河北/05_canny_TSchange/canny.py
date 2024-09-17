#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 22:40
# @Author  : zgh
# @FileName: canny.py
# @Software: PyCharm


import cv2 as cv
import numpy as np


# 1. 使用高斯滤波核进行滤波
def smooth(image, sigma, length):
	# 构建高斯卷积核
	k = length // 2
	# 高斯滤波核尺寸
	gaussian = np.zeros([length, length])
	# 分母
	coef = 2 * np.pi * sigma ** 2
	# 从（0,0）开始，卷积核中心权值最大，越边缘权值越小
	# 1 2 1
	# 2 5 2
	# 1 2 1
	for i in range(length):
		for j in range(length):
			gaussian[i, j] = np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2)) / coef
	gaussian = gaussian / np.sum(gaussian)  # 归一化

	# 进行卷积操作
	W, H = image.shape
	# 卷积后的图像尺寸（w-l+2p）/s+1
	new_image = np.zeros([W - k * 2, H - k * 2])

	for i in range(W - 2 * k):
		for j in range(H - 2 * k):
			new_image[i, j] = np.sum(image[i:i + length, j:j + length] * gaussian)

	new_image = np.uint8(new_image)
	return new_image


# 2. 使用Sobel算子计算梯度
def get_gradients(image):
	# 构建sobel算子
	Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 竖直方向
	Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # 水平方向
	W, H = image.shape
	gradients = np.zeros([W - 2, H - 2])
	theta = np.zeros([W - 2, H - 2])
	for i in range(W - 2):
		for j in range(H - 2):
			# k=3 计算竖直方向
			dx = np.sum(image[i:i + 3, j:j + 3] * Sx)
			# k=3 计算水平方向
			dy = np.sum(image[i:i + 3, j:j + 3] * Sy)
			# 梯度的模长
			gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2)
			# 梯度方向
			if dx == 0:
				theta[i, j] = np.pi / 2
			else:
				theta[i, j] = np.arctan(dy / dx)
	gradients = np.uint8(gradients)
	return gradients, theta


#  3. NMS非极大值抑制
def NMS(gradients, direction):
	W, H = gradients.shape
	# 非最大化抑制图像初始化 即为原始梯度值矩阵除去4个边缘
	nms = np.copy(gradients[1:-1, 1:-1])

	for i in range(1, W - 1):
		for j in range(1, H - 1):
			# 方向角
			theta = direction[i, j]
			# 对角度进行tan得到 dy/dx
			k = np.tan(theta)
			# 在各个方向上，保留梯度最大的点，若通过差值得到的点的梯度大于该点，则该点删除。
			# 45-90,225-270度之间 dy>dx
			# 得到d[i,j]方向上梯度最大的值
			if theta > np.pi / 4:
				k = 1 / k  # = dx/dy
				d1 = gradients[i - 1, j] * (1 - k) + gradients[i - 1, j + 1] * k
				d2 = gradients[i + 1, j] * (1 - k) + gradients[i + 1, j - 1] * k

			# 0-45,180-225度之间 dx>dy
			elif theta >= 0:
				d1 = gradients[i, j - 1] * (1 - k) + gradients[i + 1, j - 1] * k
				d2 = gradients[i, j + 1] * (1 - k) + gradients[i - 1, j + 1] * k

			# 135-180 315-360之间 dy<dx
			elif theta >= - np.pi / 4:
				k *= -1
				d1 = gradients[i, j - 1] * (1 - k) + gradients[i - 1, j - 1] * k
				d2 = gradients[i, j + 1] * (1 - k) + gradients[i + 1, j + 1] * k

				# 90-135 270-315之间 dy》dx
			else:
				k = -1 / k
				d1 = gradients[i - 1, j] * (1 - k) + gradients[i - 1, j - 1] * k
				d2 = gradients[i + 1, j] * (1 - k) + gradients[i + 1, j + 1] * k
			# 如果存在gradients[i, j]周围点中心梯度大于该点，则该店梯度置0
			if d1 > gradients[i, j] or d2 > gradients[i, j]:
				nms[i - 1, j - 1] = 0
	return nms


# 4.双阈值操作 返回二值图
# 获取并且保留高阈值的所有边，并且像素置255。其次获取低阈值的边，仅仅保留与高阈值边相连接的边，并且像素置255
# k[i,j]>maxVal------>255
# k[i,j]周围
def thresholding(nms, minVal, maxVal):
	# 初始化标记矩阵
	vis = np.zeros_like(nms)
	# 初始化结果矩阵
	edge = np.zeros_like(nms)
	W, H = edge.shape

	# 对一个像素周围的8个像素进行检测
	def check(i, j):
		# 如果检测到边缘则返回
		if (i >= W or i < 0 or j >= H or j < 0 or vis[i, j] == 1):
			return
		# 对于检测过的点，标记为1
		vis[i, j] = 1
		# 这个点大于低阈值，置255
		if nms[i, j] >= minVal:
			edge[i, j] = 255

	# 对每个像素点进行检测
	for w in range(W):
		for h in range(H):
			# 标记为1检测过了
			if vis[w, h] == 1:
				continue
			# 小于小阈值，修改标记
			elif nms[w, h] <= minVal:
				vis[w, h] = 1
			# 大于阈值，置255并且周围8个点进行检测
			elif nms[w, h] >= maxVal:
				vis[w, h] = 1
				edge[w, h] = 255  # sure-edge
				check(w - 1, h - 1)
				check(w - 1, h)
				check(w - 1, h + 1)
				check(w, h - 1)
				check(w, h + 1)
				check(w + 1, h - 1)
				check(w + 1, h)
				check(w + 1, h + 1)
	return edge


# main


def main():
	image = cv.imread("lane.jpg", 0)
	smoothed_image = smooth(image, sigma=2, length=3)
	gradients, direction = get_gradients(smoothed_image)
	nms = NMS(gradients, direction)
	edge = thresholding(nms, 30, 50)
	# cv.imshow("smoothed_image", smoothed_image)
	# cv.imshow("gradients", gradients)
	# cv.imshow("nms", nms)
	cv.imshow("edge", edge)
	# cv.imwrite('gradients.jpg', gradients, [cv.IMWRITE_PNG_COMPRESSION, 0])
	cv.waitKey(0)
	cv.destroyAllWindows()


if __name__ == '__main__':
	main()
