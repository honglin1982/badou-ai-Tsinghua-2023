import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def GussianNoise(img, sigma, mean, percentage):
    #获得图像尺寸
    width, height = img.size
    #随机噪声数量
    noise_num = int(percentage * width * height)
    #读取原图数据
    data = list(img.getdata())
    data = np.array(data).reshape(width, height, 3)

    for i in range(noise_num):
        #随机选取位置
        randX = random.randint(0, width - 1)
        randY = random.randint(0, height - 1)
        #加噪
        data[randX][randY] = data[randX][randY] + random.gauss(mean, sigma)
        #防止越界
        for index, j in enumerate(data[randX][randY]):
            if j < 0:
                data[randX][randY][index] = 0
            if j > 255:
                data[randX][randY][index] = 255
    return data


if __name__ == "__main__":
    filename = 'lenna.png'
    img = Image.open(filename)

    sigma = 4
    mean = 2
    percentage = 0.8

    bilinear_im = GussianNoise(img,2,4,0.8)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(bilinear_im)
    plt.show()