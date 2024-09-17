import random


import cv2 as cv
import numpy as np


# 添加高斯噪声，但是由于随机点可能相同，噪声比例会低于需求比例
def gauss_noise(img, sigma, mean, p):
	# 注意这里需要是哟个copy拷贝一份进行处理，不然之后原始图形会变化
	noise_img = np.copy(img)
	w, h = img.shape
	# 记录那些点添加过，用于计算真实的噪声比例
	index_map = np.zeros_like(img)
	loss = 0
	num_of_p = int(w * h * p)
	for i in range(num_of_p):
		random_x = random.randint(0, w - 1)
		random_y = random.randint(0, h - 1)
		if index_map[random_x, random_y] == 0:
			noise_img[random_x, random_y] += random.gauss(mean, sigma)
			if noise_img[random_x, random_y] > 255:
				noise_img[random_x, random_y] = 255
			elif noise_img[random_x, random_y] < 0:
				noise_img[random_x, random_y] = 0
			index_map[random_x, random_y] = 1
		else:
			# 重复的点不进行噪声添加，但是记录数目
			loss += 1
	num_of_p -= loss
	# 真实的比例
	p_t = round(num_of_p / (w * h), 2)
	return noise_img, p_t


# 第二种添加高斯噪声的方法，使用set容器，获取满足比例的噪声点。
def guass_noise_true(img, sigma, mean, p):
	noise_img = np.copy(img)
	w, h = img.shape
	num_of_p = int(w * h * p)
	# 初始化候选点集合
	ps = set()
	num_p_true = 0
	# 添加候选点，直到满足比例要求数目
	while len(ps) < num_of_p:
		x = random.randint(0, w - 1)
		y = random.randint(0, h - 1)
		ps.add((x, y))
	# 遍历候选点，添加高斯噪声
	for p in ps:
		random_x, random_y = p
		noise_img[random_x, random_y] += random.gauss(mean, sigma)
		if noise_img[random_x, random_y] > 255:
			noise_img[random_x, random_y] = 255
		elif noise_img[random_x, random_y] < 0:
			noise_img[random_x, random_y] = 0
		num_p_true += 1
	# 计算添加比例
	p_t = round(num_p_true / (w * h), 2)
	return noise_img, p_t


def jy_noise(img, snr):
	noise_img = np.copy(img)
	w, h = img.shape
	num_of_p = int(w * h * snr)
	# 初始化候选点集合
	ps = set()
	num_p_true = 0
	# 添加候选点，直到满足比例要求数目
	while len(ps) < num_of_p:
		x = random.randint(0, w - 1)
		y = random.randint(0, h - 1)
		ps.add((x, y))
	# 遍历候选点，添加高斯噪声
	for p in ps:
		random_x, random_y = p
		if random.random() < 0.5:
			noise_img[random_x, random_y] = 0
		else:
			noise_img[random_x, random_y] = 255
		num_p_true += 1
	# 计算添加比例
	p_t = round(num_p_true / (w * h), 2)
	return noise_img, p_t


def test(img, noise_name, p):
	sigma = 2
	mean = 4
	img_noise = np.zeros_like(img)
	p_t = 0
	if noise_name == "guass":
		print("高斯噪声，有损失版本\n")
		img_noise, p_t = gauss_noise(img, 2, 4, p)
	elif noise_name == "guass_ture":
		print("高斯噪声，无损失版本\n")
		img_noise, p_t = guass_noise_true(img, sigma, mean, p)
	elif noise_name == "jy_noise":
		print("椒盐噪声，无损失版本\n")
		img_noise, p_t = jy_noise(img, p)
	else:
		print("噪声类型不存在")
	return img_noise, p_t


def main():
	img = cv.imread("lane.jpg", 0)
	p = 0.2
	noise_img, p_t = test(img, "guass", p)
	print("原始噪声比例为{}\n真实的噪声比例为{}".format(p, p_t))
	cv.imshow("img", img)
	cv.imshow("noise_img", noise_img)
	cv.waitKey(0)


if __name__ == '__main__':
	main()
