
import cv2
import numpy as np
import random

def hanming(hash1, hash2):
    if len(hash1) != len(hash2):
        return
    n = len(hash1)
    ans = 0
    for i in range(n):
        if hash1[i] != hash2[i]:
            ans += 1
    return ans


def mean_hash(gray):
    '''
    均值哈希
    :return:
    '''
    resize = cv2.resize(gray, (8, 8))
    ans = []
    total = 0
    for i in range(8):
        for j in range(8):
            total += resize[i][j]
    mean_val = total / 64
    for i in range(8):
        for j in range(8):
            val = 1 if resize[i][j] > mean_val else 0
            ans.append(val)
    return ans


def dHash(gray):
    '''
    差值哈希
    :return:
    '''
    resize = cv2.resize(gray, (9, 8))
    ans = []
    for i in range(8):
        for j in range(1, 9):
            val = 1 if resize[i][j] > resize[i][j-1] else 0
            ans.append(val)
    return ans


def random_gaussion_noise(img, mean, sigma, percentage):
    '''
    高斯噪声
    :param img:
    :param mean:
    :param sigma:
    :return:
    '''
    w = img.shape[0]
    h = img.shape[1]
    new_img = img
    random_cnt = int(w * h * percentage)
    for i in range(random_cnt):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        pixel = new_img[x][y] + random.gauss(mean, sigma) * np.random.randint(-1, 2)

        if pixel > 255:
            pixel = 255
        if pixel < 0:
            pixel = 0

        new_img[x][y] = pixel

    return new_img

def salt_pepper_noise(img, percentage):
    '''
    椒盐噪声
    :param img:
    :param percentage:
    :return:
    '''
    new_img = img.copy() # 此处需要深度拷贝，否则会修改原图，影响后续调用的结果
    w = img.shape[0]
    h = img.shape[1]
    noise_cnt = int(w * h * percentage)
    for i in range(noise_cnt):
        print(i)
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        ran = np.random.random()
        if ran >= 0.5:
            new_img[x][y] = 255
        else:
            new_img[x][y] = 0
    return new_img




if __name__ == '__main__':
    ori = cv2.imread('lenna.png')
    gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
    # 生成噪声图片
    noise = salt_pepper_noise(gray, 0.5)
    cv2.imwrite('noise.png', noise)
    noise = cv2.imread('noise.png')
    noise_gray = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    # 均值哈希
    ori_mean_hash = mean_hash(gray)
    noise_mean_hash = mean_hash(noise_gray)
    print('ori_mean_hash:{}'.format(ori_mean_hash))
    print('noise_mean_hash:{}'.format(noise_mean_hash))
    print('均值哈希汉明距离为：{}'.format(hanming(ori_mean_hash, noise_mean_hash)))
    # 差值哈希
    ori_dHash = dHash(gray)
    noise_dHash = dHash(noise_gray)
    print('ori_dHash:{}'.format(ori_dHash))
    print('noise_dHash:{}'.format(noise_dHash))
    print('差值哈希汉明距离为：{}'.format(hanming(ori_dHash, ori_dHash)))
