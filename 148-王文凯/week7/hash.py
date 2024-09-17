import cv2
import numpy as np
import random

# 均值哈希
def average_hash(img):
    # step_1 缩放图片为8 * 8，保留结构，除去细节
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # step_2 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s, hash_res = 0, ''
    # step_3 求平均值
    for i in range(8):
        for j in range(8):
            s += img_gray[i, j]
    avg = s / 64
    # step_4 比较 像素值大于平均值记为1，否则记为0
    # step_5 根据生成的1，0生成hash
    for i in range(8):
        for j in range(8):
            hash_res += '1' if img_gray[i, j] > avg else '0'
    return hash_res

# 差值哈希
def difference_hash(img):
    # step_1 缩放图片为8 * 9，保留结构，除去细节
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    # step_2 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # step_3 像素值大于后一个像素值记作1，相反记作0，生成哈希
    hash_res = ''
    for i in range(8):
        for j in range(8):
            hash_res += '1' if img_gray[i, j] > img_gray[i, j + 1] else '0'
    return hash_res

# 噪声图片生成
def img_generate_noise(img):
    h, w = img.shape[:2]
    noise_img, noise_num = np.copy(img), int(0.1 * h * w)
    for i in range(noise_num):
        rand_x, rand_y = random.randint(0, h - 1), random.randint(0, w - 1)
        noise_img[rand_x, rand_y] = 0 if random.random() <= 0.5 else 255
    return noise_img

# hash比较
def hash_compare(hash_1,hash_2):
    n = 0
    if len(hash_1) != len(hash_2):
        return -1
    for i in range(len(hash_1)):
        n += 1 if hash_1[i] != hash_2[i] else 0
    return n

def main():
    img = cv2.imread('./img/lenna.png')
    img_noise = img_generate_noise(img)

    img_average_hash = average_hash(img)
    img_noise_average_hash = average_hash(img_noise)

    print('原图均值哈希值：', img_average_hash)
    print('噪声图图均值哈希值：', img_noise_average_hash)
    print('均值哈希相似度：', hash_compare(img_average_hash, img_noise_average_hash))

    img_difference_hash = difference_hash(img)
    img_noise_difference_hash = difference_hash(img_noise)

    print('原图均值哈希值：', img_difference_hash)
    print('噪声图图均值哈希值：', img_noise_difference_hash)
    print('差值哈希相似度：', hash_compare(img_difference_hash, img_noise_difference_hash))

if __name__ == '__main__':
    main()
