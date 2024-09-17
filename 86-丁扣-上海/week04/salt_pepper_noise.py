import cv2
import random

import numpy as np


def salt_pepper_noise_func(img_src: np.ndarray, percent):
    """
    椒盐噪声
    含义：黑白点
    """
    new_img = img_src.copy()
    noise_number = int(img_src.shape[0] * img_src.shape[1] * percent)
    for i in range(noise_number):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 椒盐噪声图片边缘不处理，故-1
        rand_x = random.randint(0, new_img.shape[0] - 1)
        rand_y = random.randint(0, new_img.shape[1] - 1)
        if random.random() < 0.5:
            new_img[rand_x, rand_y] = 0
        else:
            new_img[rand_x, rand_y] = 255

    return new_img


if __name__ == '__main__':
    img = cv2.imread(r'../file/lenna.png')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = salt_pepper_noise_func(img_gray, percent=0.1)
    cv2.imshow(" src ", img_gray)
    cv2.imshow(" salt_pepper ", img_gaussian)
    cv2.waitKey(10000)
    pass

