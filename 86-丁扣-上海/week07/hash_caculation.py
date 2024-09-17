import cv2
import numpy as np


def hash_mean(img: np.ndarray):
    """ 均值hash """
    new_img = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 求平均灰度
    mean_v = np.mean(img_gray)
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > mean_v:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


def hash_sub(img: np.ndarray):
    """ 差值哈希 """
    new_img = cv2.resize(img, dsize=(9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


def hash_match(h1: str, h2: str):
    # hash长度不同则返回-1代表传参出错
    if len(h1) != len(h2):
        return -1

    # 遍历判断
    n = 0
    for i in range(len(h1)):
        # 不相等则n计数+1，n最终为相似度
        if h1[i] != h2[i]:
            n = n + 1
    return n


if __name__ == '__main__':
    path = f'../file/lenna.png'
    h1 = hash_mean(cv2.imread(path))
    print(h1)
    h2 = hash_sub(cv2.imread(path))
    print(h2)
    print(hash_match(h1, h2))
    pass

