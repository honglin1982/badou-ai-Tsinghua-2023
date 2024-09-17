import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img = cv2.imread("lenna.png")

#BGR转RGB
img_rgb1 = img[:,:,[2,1,0]]
img_rgb2 = img[:,:,::-1]
img_rgb3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(331)
plt.imshow(img_rgb1)
plt.subplot(332)
plt.imshow(img_rgb2)
plt.subplot(333)
plt.imshow(img_rgb3)

#灰度化
h, w = img.shape[:2]
img_gray1 = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray1[i, j] = int(m[0] * 0.59 + m[1] * 0.11 + m[2] * 0.3)
plt.subplot(334)
plt.imshow(img_gray1, cmap='gray')

img_gray2 = rgb2gray(img)
plt.subplot(335)
plt.imshow(img_gray2, cmap='gray')

img_gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(336)
plt.imshow(img_gray3, cmap='gray')

#二值化
gray_h, gray_w = img_gray2.shape
img_binary1 = img_gray2
for i in range(gray_h):
    for j in range(gray_w):
        if img_gray2[i, j] > 0.5:
            img_binary1[i, j] = 1
        else:
            img_binary1[i, j] = 0
plt.subplot(337)
plt.imshow(img_binary1, cmap='gray')

img_binary2 = np.where(img_gray2 > 0.5, 1, 0)
plt.subplot(338)
plt.imshow(img_binary2, cmap='gray')


plt.show()
