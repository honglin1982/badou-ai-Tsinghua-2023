import cv2
import random
import numpy as np


def PepperSaltNoise(src, percentage):
    # 按比例计算需要进行噪声处理的像素点个数，注意取整数据
    noiseNum = int(src.shape[0] * src.shape[1] * percentage)

    for i in range(0, noiseNum):
        # 随机取x,y坐标
        tmpx = random.randint(0, src.shape[0] - 1)
        tmpy = random.randint(0, src.shape[1] - 1)

        # 根据0到1的随机数，确定该坐标对应的像素值是替换成椒噪声0还是盐噪声255
        src[tmpx, tmpy] = 0 if random.randint(0, 1) == 0 else 255

    return src


img = cv2.imread('lenna.png', 0)
img1 = PepperSaltNoise(img, 0.05)   # 该方法会直接改变源图
img = cv2.imread('lenna.png', 0)
cv2.imshow('PepperSalt', np.hstack([img, img1]))
cv2.waitKey()
cv2.destroyAllWindows()
