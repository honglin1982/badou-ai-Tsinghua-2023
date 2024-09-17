import cv2
import numpy as np


def resize_image(image, size):
    # 使用列表推导式遍历并调整图像大小
    resized_images = [cv2.resize(img, size) for img in image]
    resized_images = np.array(resized_images)  # 转换为 NumPy 数组
    return resized_images


def print_answer(argmax):
    with open("./data/index_word.txt", "r", encoding='utf-8') as f:
        synset = [i.split(";")[1][:-1] for i in f.readlines()]
    return synset[argmax]
