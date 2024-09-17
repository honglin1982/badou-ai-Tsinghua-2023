import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
实现sift(调用)
实现最小二乘法和ransac
实现差值和均值哈希
'''

# 最小二乘法
def least_square(data):
    '''
    (kx + b - y) ^ 2对k、n求偏导：
        Σ kxx + bx - xy = 0
        Σ kx + b - y = 0
    设x0, y0分别是x,y的均值
    由2式得 b = y0 - k * x0
    代入1式得
        Σ kxx + (y0 - k * x0)x - xy = 0
    即   k = Σ (xy - y0*x) / (xx - x0x)
    '''
    x = data[:, 0]
    y = data[:, 1]
    n = data.shape[0]
    x0, y0 = np.sum(x) / n, np.sum(y) / n
    result_k = (np.dot(x, y) - y0 * np.sum(x)) / (np.dot(x, x) - x0 * np.sum(x))
    result_b = y0 - result_k * x0
    return result_k, result_b


data = np.array([[1, 6], [2, 5], [3, 7], [4, 10]])
k, b = least_square(data)
print("k = {}, b = {}".format(k, b))


# ransac
def ransac(data, iter, random_rate, threshold, good_rate):
    best_k, best_b, best_num = 0, 0, 0
    data_total_num = data.shape[0]
    random_num = int(random_rate * data_total_num)
    good_num = int(good_rate * data_total_num)
    for i in range(iter):
        shuffle_index = np.arange(data_total_num)
        np.random.shuffle(shuffle_index)
        in_group, other_data = data[shuffle_index[:random_num]], data[shuffle_index[random_num:]]
        k_tmp, b_tmp = least_square(in_group)
        good_tmp = 0
        for e in other_data:
            if -threshold < k_tmp * e[0] + b_tmp - e[1] < threshold:
                good_tmp += 1
        if good_tmp > best_num:
            best_k, best_b, best_num = k_tmp, b_tmp, good_tmp
        if best_num >= good_num:
            break
    if best_num >= good_num:
        return best_k, best_b
    else:
        print("ransac没有找到符合要求的解，当前点符合内群的阈值：{}，外群在阈值内的点数量：{}".format(threshold, best_num))
        return best_k, best_b


sample_num = 100
error_num = 40
k_real = 10
b_real = 5
data_x = np.random.normal(size=(sample_num, 1)) * 10
data_y = k_real * data_x + b_real

error_indexs = np.arange(sample_num)
np.random.shuffle(error_indexs)
error_indexs = error_indexs[:error_num]

data_x_noisy = data_x + np.random.normal(size=(sample_num, 1))
data_y_noisy = data_y + np.random.normal(size=(sample_num, 1))
for error_index in error_indexs:
    data_y_noisy[error_index] = 60 * np.random.random()

data = np.hstack((data_x_noisy, data_y_noisy))

k_line, b_line = least_square(data)
k_ransac, b_ransac = ransac(data, 1000, 0.1, 10, 0.4)

plt.scatter(data_x_noisy, data_y_noisy, label="data")
plt.plot(data_x,  k_real * data_x + b_real, label="real_line", color="r")
plt.plot(data_x_noisy,  k_line * data_x_noisy + b_line, label="least_square_line", color="y")
plt.plot(data_x_noisy,  k_ransac * data_x_noisy + b_ransac, label="ransac_line", color="k")
plt.legend()
plt.show()



# sift(调用)
def sift_call(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoiots, descriptions = sift.detectAndCompute(img, None)
    return keypoiots, descriptions


img = cv2.imread("lenna.png")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kps, descripts = sift_call(img)
cv2.drawKeypoints(img, keypoints=kps, outImage=img, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT, color=(0, 0, 255))
cv2.imshow("img", img)
cv2.waitKey(0)



# 差值哈希和均值哈希
def mean_hash(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(8, 8), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg = np.sum(img) / 64
    hash_res = ""
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hash_res += "1" if img[i, j] > avg else "0"
    return hash_res

def diff_hash(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(9, 8), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_res = ""
    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 1):
            hash_res += "1" if img[i, j] > img[i, j+1] else "0"
    return hash_res


print("mean_hash: ", mean_hash("lenna.png"))
print("diff_hash: ", diff_hash("lenna.png"))