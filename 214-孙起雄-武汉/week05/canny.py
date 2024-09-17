# 实现canny算法
import cv2
import numpy as np


def canny_edge_detection(image, low_threshold, high_threshold):
    # 1. 转换图像为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 使用高斯滤波平滑图像
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3. 计算图像中的梯度信息
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # 4. 计算边缘的梯度强度和方向
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi

    # 5. 应用非极大值抑制，细化边缘
    suppressed_image = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            angle = gradient_direction[i, j]

            # 根据梯度方向进行非极大值抑制
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
                if (gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1]) and (
                        gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
            elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j - 1]) and (
                        gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j + 1]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
            elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j]) and (
                        gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
            elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j + 1]) and (
                        gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j - 1]):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

    # 6. 使用双阈值进行边缘检测和边缘连接
    edges = np.zeros_like(suppressed_image)
    strong_edges = (suppressed_image > high_threshold)
    weak_edges = (suppressed_image >= low_threshold) & (suppressed_image <= high_threshold)

    # 将强边缘直接标记为边缘
    edges[strong_edges] = 255

    # 对弱边缘进行边缘连接
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j]:
                if (strong_edges[i - 1:i + 2, j - 1:j + 2]).any():
                    edges[i, j] = 255

    # 7. 返回边缘图像
    return edges

    # 读取图像
    image = cv2.imread('image.jpg')

    # 设置低阈值和高阈值
    low_threshold = 50
    high_threshold = 150

    # 调用canny_edge_detection函数进行边缘检测
    edges = canny_edge_detection(image, low_threshold, high_threshold)

    # 显示边缘图像
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


