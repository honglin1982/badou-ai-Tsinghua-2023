import random
import cv2


# 椒盐噪声的函数参数：图，噪点百分比
def pepper(src, per):
    img = src
    noise_nums = int(img.shape[0] * img.shape[1] * per)
    # 对每个噪点取随机位置，并将该点随机变成0或255
    for i in range(noise_nums):
        randx = random.randint(0, img.shape[0]-1)
        randy = random.randint(0, img.shape[1]-1)
        if random.random() <= 0.5:
            img[randx, randy] = 0
        else:
            img[randx, randy] = 255
    return img


# 取一张图的灰度图（imread操作即可）变成椒盐模糊图，并与灰度图（BGR2GRAY）做对比
img1 = cv2.imread('lenna.png', 0)
img_Pepper = pepper(img1, 0.2)
img2 = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('show gray', img_gray)
cv2.imshow('pepper', img_Pepper)
cv2.waitKey()
