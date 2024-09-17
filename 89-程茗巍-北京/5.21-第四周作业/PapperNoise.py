import numpy as np
import cv2
import random

def salt_papper_noise(image, slat_prob, pepper_prob):
    nosiy_image = np.copy(image)
    #获得图像总像素数和形状
    num_piexls = np.prod(image.shape[:2])
    width = image.shape[0]
    height = image.shape[1]

    #计算添加的椒盐数量
    num_slat = np.ceil(num_piexls * slat_prob)
    num_pepper = np.ceil(num_piexls * pepper_prob)

    #随机加噪
    for i in range(int(num_slat)):
        #随机选取位置
        randX = random.randint(0, width - 1)
        randY = random.randint(0, height - 1)
        #加噪
        nosiy_image[randX][randY] = 255

    for i in range(int(num_pepper)):
        #随机选取位置
        randX = random.randint(0, width - 1)
        randY = random.randint(0, height - 1)
        #加噪
        nosiy_image[randX][randY] = 0


    return nosiy_image


if __name__ == "__main__":
    image = cv2.imread('lenna.png')
    # 添加椒盐噪声，盐噪声比例0.02，椒噪声比例0.02
    salt_prob = 0.2
    pepper_prob = 0.2
    noisy_image = salt_papper_noise(image, salt_prob, pepper_prob)
    # 显示原始图片和有噪声的图片
    cv2.imshow('original', image)
    cv2.imshow('with salt and pepper noise', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
