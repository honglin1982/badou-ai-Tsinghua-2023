import cv2

img = cv2.imread('../lenna.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个sift对象
sift = cv2.xfeatures2d.SIFT_create()
# 通过detectAndCompute方法得出关键点和描述

'''
kp, des = sift.detectAndCompute(image, mask, useProvidedKeypoints=False)
image：需要进行特征提取的源图像，可以是灰度图像或彩色图像，
       数据类型为 numpy.ndarray，通常使用 cv2.imread 函数读取；
mask：掩码图像，只有与掩码图像对应位置为白色（255）的像素点才会被考虑，其他像素点将被忽略，
      数据类型同样为 numpy.ndarray；
useProvidedKeypoints：可选参数，布尔类型，是否使用用户提供的关键点，如果为 True，
      则需要在 kp 参数中提供关键点，否则函数将自动检测关键点，并返回关键点和特征描述符。
'''
keypoints, descriptor = sift.detectAndCompute(gray, None)

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
cv2.drawKeypoints(img, keypoints, img, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('sift keypoints', img)
cv2.waitKey(0)
