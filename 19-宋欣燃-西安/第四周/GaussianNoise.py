import cv2
import random
# 1 代码GaussianNoise只能处理灰度图情况，函数内部没对彩色图做判断
# 2 加入高斯噪声的时候不一定要把RGB图像转成灰度图处理，可以分离通道添加噪声后再合成
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0, src.shape[0]-1)
        randY=random.randint(0, src.shape[1]-1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY]+random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg
# 1：加载彩色图片 0：加载灰度模式
img = cv2.imread('/Users/songxinran/Documents/GitHub/badou-AI-Tsinghua-2023/lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)

img = cv2.imread('/Users/songxinran/Documents/GitHub/badou-AI-Tsinghua-2023/lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)

