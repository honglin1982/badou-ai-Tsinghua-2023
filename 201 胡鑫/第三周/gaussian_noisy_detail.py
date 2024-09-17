import cv2
import random

def gaussian_noisy_rgb(img, mean, sigma, percentage):
    noisy_img = img
    h, w, channels = img.shape
    # 计算随机坐标个数
    num = int(h * w * percentage)
    for _ in range(num):
        # 获取随机坐标, 默认图像边框不处理, 所以有-1
        randx = random.randint(0, h-1)
        randy = random.randint(0, w-1)
        for c in range(channels):
            # 对于每个通道, 有三分之一的概率进行高斯噪声的添加
            if random.random() <= (1/3):
                noisy_img[randx, randy, c] += random.gauss(mean, sigma)
                # 图像灰度值的边界处理
                if noisy_img[randx, randy, c] > 255:
                    noisy_img[randx, randy, c] = 255
                elif noisy_img[randx, randy, c] < 0:
                    noisy_img[randx, randy, c] = 0
    return noisy_img

def gaussian_noisy(img, mean, sigma, percentage):
    noisy_img = img
    h, w = img.shape[:2]
    # 如果为rgb图像, 使用gaussian_noisy_rgb()
    if len(img.shape) == 3:
        return gaussian_noisy_rgb(img, mean, sigma, percentage)
    # 计算随机坐标个数    
    num = int(h * w * percentage)
    for _ in range(num):
        # 随机获取坐标, 图像的边框不处理, 所以有-1
        randx = random.randint(0, h-1)
        randy = random.randint(0, w-1)
        # 利用random.gauss(均值, 标准差)添加服从高斯分布的噪声
        noisy_img[randx, randy] += random.gauss(mean, sigma)
        # 灰度值边界处理
        if noisy_img[randx, randy] > 255:
            noisy_img[randx, randy] = 255
        elif noisy_img[randx, randy] < 0:
            noisy_img[randx, randy] = 0
    return noisy_img

def test_gray():
    img = cv2.imread('../lenna.png', 0)
    img1 = gaussian_noisy(img, 2, 4, 0.8)

    img = cv2.imread('../lenna.png', 0)
    cv2.imshow('source', img)
    cv2.imshow('gaussian', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_rgb():
    img = cv2.imread('../lenna.png')
    img1 = gaussian_noisy(img, 2, 4, 0.8)
    
    img = cv2.imread('../lenna.png')
    cv2.imshow('source', img)
    cv2.imshow('gaussian', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # test_gray()
    test_rgb()
   
    