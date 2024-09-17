import random
import cv2

# 高斯噪声的函数参数：图，sigma，mean，噪点百分比
def gauss(src, sigma, mu, per):
    img = src
    noise_Nums = int(img.shape[0] * img.shape[1] * per)
    # 对每个噪点取随机位置，并为该点增加随机的高斯数，数额收缩在0到255之间
    for i in range(noise_Nums):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)
        img[randX, randY] = img[randX, randY] + random.gauss(mu, sigma)
        if img[randX, randY] < 0:
            img[randX, randY] = 0
        elif img[randX, randY] > 255:
            img[randX, randY] = 255
    return img

# 取一张图的灰度图（imread操作即可）变成高斯模糊图，并与灰度图（BGR2GRAY）做对比
img1 = cv2.imread('lenna.png', 0)
img_Gauss = gauss(img1, 2, 8, 0.8)
img2 = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('show gray', img_gray)
cv2.imshow('gauss', img_Gauss)
cv2.waitKey()
