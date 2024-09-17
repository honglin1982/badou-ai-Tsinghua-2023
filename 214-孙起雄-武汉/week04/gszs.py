# 实现高斯噪声
import cv2
import numpy as np


def add_gaussian_noise(image, mean, std_dev, noise_intensity):
    # 根据噪声强度计算噪声的尺度
    scale = noise_intensity * std_dev

    # 生成与输入图像形状相同、使用计算得到的尺度的随机高斯噪声
    noise = np.random.normal(mean, scale, image.shape)

    # 将生成的噪声添加到图像中
    noisy_image = image + noise

    # 将像素值限制在有效范围 [0, 255] 内
    noisy_image = np.clip(noisy_image, 0, 255)

    # 将像素值转换为整数类型
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image


# 读取原始图像
image = cv2.imread('7.jpg', cv2.IMREAD_GRAYSCALE)

mean = 0  # 高斯分布的均值
std_dev = 20  # 高斯分布的标准差
noise_intensity = 0.8  # 噪声强度（调整该参数以控制噪声的强度）

# 使用指定的参数添加高斯噪声
noisy_image = add_gaussian_noise(image, mean, std_dev, noise_intensity)

# 保存带噪声的图像
cv2.imwrite('noisy_image.jpg', noisy_image)

# 显示原始图像和带噪声的图像
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

