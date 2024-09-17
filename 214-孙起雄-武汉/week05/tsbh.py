# 实现透视变换算法

import cv2
import numpy as np


def perspective_transform(image, src_points, dst_points):
    # 1. 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 2. 进行透视变换
    transformed_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    # 3. 返回透视变换后的图像
    return transformed_image


# 示例用法：
# 读取图像
image = cv2.imread('7.jpg')

# 指定原始图像中的四个顶点坐标
src_points = np.float32([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])

# 指定目标图像中的四个顶点坐标，以定义透视变换后的图像形状
dst_points = np.float32([[100, 100], [500, 100], [500, 400], [100, 400]])

# 调用perspective_transform函数进行透视变换
transformed_image = perspective_transform(image, src_points, dst_points)

# 显示透视变换后的图像
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
