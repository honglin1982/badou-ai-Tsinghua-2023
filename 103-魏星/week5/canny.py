import cv2
import matplotlib.pyplot as plt


'''
1、对图像进行灰度化
2、对图像进行高斯滤波
3、检测图像中的水平、垂直和对角边缘(如Prewitt，Sobel算子等)
4、对梯度幅值进行非极大值抑制
5、用双阈值算法检测和连接边缘
'''


img = cv2.imread("flower.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny_img = cv2.Canny(img_gray, 200, 300)

plt.show()
plt.figure()

plt.subplot(121)
plt.imshow(img_gray)
plt.subplot(122)
plt.imshow(canny_img)
plt.show()




