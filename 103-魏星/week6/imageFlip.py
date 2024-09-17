# coding: utf-8
import cv2
import matplotlib.pyplot as plt

img_dir = "rabbit.png"
img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 水平翻转
flip_horizontal = cv2.flip(img, 1)
# 垂直翻转
flip_vertical = cv2.flip(img, 0)
# 水平加垂直翻转
flip_hv = cv2.flip(img, -1)


plt.show()
plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']

plt.subplot(221)
plt.imshow(img)
plt.title("src")
plt.subplot(222)
plt.imshow(flip_horizontal)
plt.title("水平翻转")
plt.subplot(223)
plt.imshow(flip_vertical)
plt.title("垂直翻转")
plt.subplot(224)
plt.imshow(flip_hv)
plt.title("水平加垂直翻转")
plt.show()

