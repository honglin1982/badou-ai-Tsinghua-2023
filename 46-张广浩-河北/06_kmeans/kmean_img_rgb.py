# -*- coding: utf-8 -*-
# @Time    : 2023/6/9 10:14
# @Author  : zgh
# @FileName: kmean_img_rgb.py
# @Software: PyCharm


# coding: utf-8
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def kmean_img_rgb(img, ks):
	imgs = [img]
	for k in ks:
		data = img.reshape((-1, 3))
		data = np.float32(data)

		criteria = (cv2.TERM_CRITERIA_EPS +
		            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

		flags = cv2.KMEANS_RANDOM_CENTERS

		compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

		centers = np.uint8(centers)
		res = centers[labels.flatten()]
		dst = res.reshape((img.shape))
		imgs.append(dst)
	return imgs


def main():
	img = cv2.imread('photo.jpg')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	ks = [2, 5, 10, 20, 50, 60, 70]
	imgs = kmean_img_rgb(img, ks)
	l_imgs = 0
	l_imgs = len(imgs) // 2 if len(imgs) % 2 == 0 else len(imgs) // 2 + 1
	new_titles = ['原始图像']+[f'聚类图像 K={k}' for k in ks]

	plt.rcParams['font.sans-serif'] = ['SimHei']

	for i in range(len(imgs)):
		plt.subplot(2, l_imgs, i+1)
		plt.imshow(imgs[i], 'gray')
		plt.title(new_titles[i])
		plt.xticks([]), plt.yticks([])

	plt.show()


if __name__ == '__main__':
	main()
