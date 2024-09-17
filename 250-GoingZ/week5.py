import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math

# canny-detail
def read_img_gray():
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def img_filter(filter, gray_img):
    f_size = filter.shape[0]
    img = np.zeros(gray_img.shape)
    for i in range(gray_img.shape[0] - f_size + 1):
        for j in range(gray_img.shape[1] - f_size + 1):
            img[i, j] = np.sum(gray_img[i:i+f_size, j:j+f_size] * filter)
    
    # img = np.uint8(img) # 浮点数直接显示会得到一张纯白图片
    return img

def Gaussian_filter(img):
    Gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    Gaussian_filter = Gaussian_filter / 16

    Gaussian_filter2 = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])
    Gaussian_filter2 = Gaussian_filter2 / 273

    print(Gaussian_filter)
    img = img_filter(Gaussian_filter2, img)
    return img

def sobel_filter(img):
    sobel_filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_x = img_filter(sobel_filter_x, img)
    img_y = img_filter(sobel_filter_y, img)

    img = (img_x + img_y) / 2
    # img = np.sqrt(img_x ** 2 + img_y ** 2)
    
    # img = np.uint8(img)
    return img

def non_max_suppression(img):
    nms_img = np.zeros(img.shape)
    delta_img = np.zeros(img.shape)
    angle = np.zeros(img.shape)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            delta_cur_x = img[i + 1, j] - img[i - 1, j]
            delta_cur_y = img[i, j + 1] - img[i, j - 1]
            delta_cur = (delta_cur_x ** 2 + delta_cur_y ** 2) ** 0.5
            delta_img[i, j] = delta_cur
            if delta_cur_x == 0:
                angle[i, j] = 90
            else:
                angle[i, j] = math.atan(delta_cur_y / delta_cur_x) * 180 / math.pi

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[0]-1):
            flag = True
            temp = delta_img[i-1 : i+2, j-1 : j+2]
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (delta_img[i, j] > num_1 and delta_img[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (delta_img[i, j] > num_1 and delta_img[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (delta_img[i, j] > num_1 and delta_img[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (delta_img[i, j] > num_1 and delta_img[i, j] > num_2):
                    flag = False
            if flag:
                nms_img[i, j] = delta_img[i, j]
    return nms_img

def dual_threshold(img):
    high_threshold = 4 * np.mean(img)
    low_threshold = 0.5 * np.mean(img)
    img = np.where(img > high_threshold, 255, 0)
    return img

def study(img):
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列
    n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2)) 
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    Gaussian_filter = np.round(Gaussian_filter, decimals=2)  # 保留两位小数

    img = img_filter(Gaussian_filter, img)
    return img 

def canny_edge_detection(img):
    img = Gaussian_filter(img)
    img = sobel_filter(img)
    img = non_max_suppression(img)
    img = dual_threshold(img)
    img = np.uint8(img)
    return img

def Perspective_transformation(img):
    src_point = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst_point = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    m = cv2.getPerspectiveTransform(src_point, dst_point)

    result = cv2.warpPerspective(img, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)

if __name__ == '__main__':
    # study()
    img = read_img_gray()
    img = canny_edge_detection(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    img = read_img_gray()
    Perspective_transformation(img)
    