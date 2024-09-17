import cv2
import random


def salt_pepper_noisy_rgb(img, percentage):
    noisy_img = img
    h, w, channels = img.shape
    # 计算随机坐标个数
    num = int(h * w * percentage)
    for _ in range(num):
        # 获取随机坐标
        randx = random.randint(0, h-1)
        randy = random.randint(0, w-1)

        for c in range(channels):
            # 对于每个通道, 有三分之一概率对其进行椒盐加噪处理
            if random.random() <= (1/3):
                # 对于需要加噪的通道的每个随机坐标的灰度值, 有1/2为0, 1/2为255
                if random.random() <= (1/2):
                    noisy_img[randx, randy, c] = 0
                else:
                    noisy_img[randx, randy, c] = 255
    return noisy_img

def salt_pepper_noisy(img, percentage):
    noisy_img = img
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        # 如果图像为rgb图像, 则使用salt_pepper_noisy_rgb()
        return salt_pepper_noisy_rgb(img, percentage)
    # 计算随机坐标个数
    num = int(h * w * percentage)
    for _ in range(num):
        # 获取随机坐标, 默认图像边缘不处理, 所以有-1
        randx = random.randint(0, h-1)
        randy = random.randint(0, w-1)
        # 对于每个随即坐标的噪音添加, 分别有二分之一概率为椒; 盐噪声
        if random.random() <= (1/2):
            noisy_img[randx, randy] = 0
        else:
            noisy_img[randx, randy] = 255
    return noisy_img

def test_gray():
    img = cv2.imread('../lenna.png', 0)
    img1 = salt_pepper_noisy(img, 0.2)

    img = cv2.imread('../lenna.png', 0)
    cv2.imshow('source', img)
    cv2.imshow('saltpepper', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_rgb():
    img = cv2.imread('../lenna.png')
    img1 = salt_pepper_noisy(img, 0.2)

    img = cv2.imread('../lenna.png')
    cv2.imshow('source', img)
    cv2.imshow('saltpepper', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # test_gray() 
    test_rgb()   