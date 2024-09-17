import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


'''
实现高斯噪声和椒盐噪声
实现PCA
'''


def GaussNoise(img, mean=0, sigma=5):
    h, w, c = img.shape
    dst_img = np.zeros(img.shape, dtype=np.uint8)
    for channel in range(c):
        for i in range(h):
            for j in range(w):
                target_p = img[i, j, channel] + random.gauss(mean, sigma)
                target_p = min(target_p, 255)
                target_p = max(target_p, 0)
                dst_img[i, j, channel] = int(target_p)
    return dst_img


def SaltPepperNoise(img, snr=0.2):
    h, w, c = img.shape
    dst_img = np.copy(img)
    for channel in range(c):
        p_num = h * w * snr
        positions_selected = set()
        while p_num > 0:
            random_h = random.randint(0, h-1)
            random_w = random.randint(0, w-1)
            if "{}_{}".format(random_h, random_w) not in positions_selected:
                positions_selected.add("{}_{}".format(random_h, random_w))
                p_num -= 1
                dst_img[random_h, random_w, channel] = 255 if random.random() > 0.5 else 0
    return dst_img


img = cv2.imread("lenna.png")
img_gauss_noise = GaussNoise(img, sigma=30)
img_salt_pepper_noise = SaltPepperNoise(img, snr=0.2)

cv2.imshow("img_origin", img)
cv2.imshow("img_gauss_nois", img_gauss_noise)
cv2.imshow("img_salt_pepper_noise", img_salt_pepper_noise)
cv2.waitKey()
