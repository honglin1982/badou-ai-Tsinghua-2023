from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#灰度化
img=cv2.imread("lenna.png")
h,w=img.shape[:2] #获取图片的高和宽
img_gray=np.zeros([h,w],img.dtype) #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m=img[i,j]
        #取出当前high和wide中的BGR坐标
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
#将BGR坐标转化为gray坐标并赋值给新图像
print(img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 在使用该方法进行图像显示之前，应该先通过 cv2.imread() 方法加载图像。
# 同时，还应该使用 cv2.waitKey() 方法等待一段时间，直到用户关闭显示窗口才退出程序。例如：



# 二值化
rows, cols = img_gray.shape
img_gray=img_gray/255.0
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
print(img_gray)

plt.subplot(221)
plt.imshow(img_gray, cmap='gray')
plt.show()




