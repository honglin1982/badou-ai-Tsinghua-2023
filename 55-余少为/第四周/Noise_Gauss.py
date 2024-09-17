import random
import numpy as np
import cv2


def GaussianNoise(src, mean, sigma, percentage):
    # 按比例计算需要进行噪声处理的像素点个数，注意取整数据
    noiseNum = int(src.shape[0] * src.shape[1] * percentage)

    for i in range(0, noiseNum):
        # 随机取x,y坐标
        tmpx = random.randint(0, src.shape[0]-1)
        tmpy = random.randint(0, src.shape[1]-1)

        # 取随机坐标对应的像素点，加上高斯随机数
        pout = src[tmpx, tmpy] + random.gauss(mean, sigma)

        # 将值放在0到255之间
        src[tmpx, tmpy] = max(min(pout, 255), 0)

    return src


img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 10, 2, 0.9)   # 该方法会直接改变源图
img = cv2.imread('lenna.png', 0)
cv2.imshow('Noise_Gauss', np.hstack([img, img1]))
cv2.waitKey()
cv2.destroyAllWindows()
