from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
img = cv2.imread("./lenna.png")
h,w = img.shape[:2]                   #获取图片的high和wide
img_gray = np.zeros([h,w],img.dtype)  #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]                 #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)    #将BGR坐标转化为gray坐标并赋值给新图像
print("image show gray:%s" %img_gray)
cv2.imshow("image show gray:", img_gray)
'''

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")
print(img)

img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray---")
print(img_gray)