import numpy as np
import cv2
import matplotlib.pyplot as plt


# Canny算子
def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize=kernel_size)  # 边缘检测

    re = cv2.getTrackbarPos('Canny', 'demo')
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 用原始颜色添加到检测的边缘上
    if re % 20 == 0:
        b, g, r = cv2.split(dst)  # 分别提取B、G、R通道
        img_new1 = cv2.merge([r, g, b])  # 重新组合为R、G、B，防止出现plt.imshow图片发蓝的情况。
        # 滑块不能滑动太快，不然显示不全。
        plt.subplot(2, 3, (int)(re / 20 + 1)), plt.xticks([]), plt.yticks([]), plt.title(re), plt.imshow(img_new1)
        #plt.show()
    #cv2.imshow('demo', dst)
    plt.show()
    #plt.imshow(dst)


lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('boy1.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #转换彩色图像为灰度图
# cv2.imshow("yuan",gray)
# cv2.waitKey()
cv2.namedWindow('demo')

# 设置调节杠,
cv2.createTrackbar('Canny', 'demo', lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # 初始化

if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()