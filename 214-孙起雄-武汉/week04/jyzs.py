# 实现椒盐噪声
import cv2
import numpy as np


def add_saltpepper_noise(image, salt_prob, pepper_prob):
    # 创建一个与输入图像形状相同的随机矩阵
    random_matrix = np.random.rand(*image.shape)

    # 将所有小于盐概率的随机元素设置为最大像素值（白色）
    image[random_matrix < salt_prob] = 255

    # 将所有大于1减去椒概率的随机元素设置为最小像素值（黑色）
    image[random_matrix > 1 - pepper_prob] = 0

    return image


# 读取原始图像
image1 = cv2.imread('7.jpg', cv2.IMREAD_GRAYSCALE)
# 显示原始图像
cv2.imshow('OriginalImage', image1)
salt_prob = 0.02  # 盐噪声概率（控制白色像素的出现频率）
pepper_prob = 0.08  # 椒噪声概率（控制黑色像素的出现频率）
# 添加椒盐噪声
noisy_image = add_saltpepper_noise(image1, salt_prob, pepper_prob)
# 保存带噪声的图像
cv2.imwrite('jy_image.jpg', noisy_image)
# 显示原始图像和带噪声的图像
cv2.imshow('NoisyImage', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

