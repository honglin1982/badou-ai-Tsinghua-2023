'''
实现RGB2GRAY(手工实现+调接口)
实现二值化
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def read_img(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  # 不直接用cv2.imread()，解决中文路径问题


def DIY_RGB2GRAY(img):
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img = 0.11 * B + 0.59 * G + 0.3 * R
    return img.astype(np.uint8)


def DIY_BINARYZATION(img):
    img = DIY_RGB2GRAY(img)
    img = np.where(img >= 127, 1, 0)
    return img


img = read_img("lenna.png")
print("----img-----")
print(img)


plt.subplot(221)
diy_gray_img = DIY_RGB2GRAY(img)
plt.title("DIY_RGB2GRAY")
plt.imshow(diy_gray_img, cmap="gray")
print("----diy_gray_img-----")
print(diy_gray_img, diy_gray_img.shape)


plt.subplot(222)
skimage_rgb2gray = rgb2gray(img)
plt.title("SKIMAGE_RGB2GRAY")
plt.imshow(skimage_rgb2gray, cmap="gray")
print("----skimage_rgb2gray-----")
print(skimage_rgb2gray, skimage_rgb2gray.shape)  # 转成0-1之间的小数了


plt.subplot(223)
diy_binaryzation = DIY_BINARYZATION(img)
plt.title("DIY_BINARYZATION")
plt.imshow(diy_binaryzation, cmap="gray")
print("----diy_binaryzation-----")
print(diy_binaryzation, diy_binaryzation.shape)


plt.subplot(224)
img_other = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
plt.title("other")
plt.imshow(img_other, cmap="gray")

plt.show()