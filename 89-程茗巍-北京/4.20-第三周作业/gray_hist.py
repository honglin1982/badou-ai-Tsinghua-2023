import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def hist(file):
    img = Image.open(file)
    gray_img = img.convert('L')
    #计算直方图
    hist, bins = np.histogram(gray_img, 256, [0, 256])
    #计算累计直方图
    cdf = hist.cumsum()
    #计算映射
    img_size = gray_img.size[0] * gray_img.size[1]
    mapping_table = np.uint8(255 * cdf/img_size)
    #应用映射
    equalized_img = gray_img.point(mapping_table)
    return equalized_img


if __name__ == '__main__':
    filename = 'lenna.png'
    im = Image.open(filename)
    equalized_img = hist(filename)
    # 显示原始图像和处理后的图像
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(im)
    ax[1].imshow(equalized_img, cmap='gray')
    #plt.show()