from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt


plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
img_gray = rgb2gray(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("----image_binary----")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
# plt.show()