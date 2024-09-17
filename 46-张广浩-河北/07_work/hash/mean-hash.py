# -*- coding: utf-8 -*-
# @Time    : 2023/6/16 10:21
# @Author  : zgh
# @FileName: mean-hash.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_hash(img):
	img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	s = 0
	hash_str = ''
	mean = np.mean(img)
	for i in range(8):
		for j in range(8):
			if gray[i][j] > mean:
				hash_str += '1'
			else:
				hash_str += '0'
	return hash_str


def dis_hash(img):
	img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hash_str = ''
	for i in range(8):
		for j in range(8):
			if gray[i, j] > gray[i, j + 1]:
				hash_str = hash_str + '1'
			else:
				hash_str = hash_str + '0'
	return hash_str


def compare_hash(hash1, hash2):
	assert len(hash1) == len(hash2), "两个hash字符长度需要一致"
	sum_d = 0
	for x1, x2 in zip(hash1, hash2):
		if x1 != x2:
			sum_d += 1
	return sum_d


def test_mean_hash(img1, img2):
	hash_str1 = mean_hash(img1)
	hash_str2 = mean_hash(img2)
	num = compare_hash(hash_str1, hash_str2)
	print("均值哈希算法结果：")
	print("图像1哈希指纹数据",hash_str1)
	print("图像2哈希指纹数据",hash_str2)
	print("均值哈希算法相似度",num)


def test_dis_hash(img1, img2):
	hash_str1 = dis_hash(img1)
	hash_str2 = dis_hash(img2)
	num = compare_hash(hash_str1, hash_str2)
	print("差值哈希算法结果：")
	print("图像1哈希指纹数据",hash_str1)
	print("图像2哈希指纹数据",hash_str2)
	print("差值哈希算法相似度",num)


if __name__ == '__main__':
	img_1 = cv2.imread("../img/lane.jpg")
	img_2 = cv2.imread("../img/lane_blur.jpg")
	img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
	img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
	plt.imshow(np.hstack((img_1, img_2)))
	plt.show()
	test_mean_hash(img_1, img_2)
	test_dis_hash(img_1, img_2)