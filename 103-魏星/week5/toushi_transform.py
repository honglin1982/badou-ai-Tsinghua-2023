import numpy as np
import cv2
import matplotlib.pyplot as plt


'''

生成透视变换矩阵：
cv2.getPerspectiveTransform(src,dst)
src:代表输入图像的四个顶点的坐标
dst:代表输出图像的四个顶点的坐标

进行透视变换
cv2.warpPerspective：
src: 原图
M： 一个3x3的变换矩阵
dsize: 输出图像的尺寸大小, 先指定(第一个参数是)col，再指定(第二个参数是)row
'''
img = cv2.imread('test_img.png')
result3 = cv2.imread('test_img.png')

src = np.float32([[639,109],[874,252],[5,914],[259,1131]])
dst = np.float32([[0,0],[240,0],[0,768],[240,768]])
print(img.shape)
# 生成透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
print(m.shape)
result = cv2.warpPerspective(img, m, (260, 775))


plt.show()
plt.figure()

plt.subplot(121)
plt.imshow(img)
plt.title("src")

plt.subplot(122)
plt.imshow(result)
plt.title("result")
plt.show()




