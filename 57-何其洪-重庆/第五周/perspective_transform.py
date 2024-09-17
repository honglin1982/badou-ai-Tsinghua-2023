# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../resources/images/perspective.jpg')
# 源图像坐标
src = np.array([[193, 68], [833, 67], [46, 978], [1048, 948]], dtype="float32")
# 转换到目标图像的坐标
dst = np.array([[0, 0], [595, 0], [0, 842], [595, 842]], dtype="float32")
# 生成透视变换矩阵；进行透视变换
matrix = cv2.getPerspectiveTransform(src, dst)
print("变换矩阵:", matrix)
result = cv2.warpPerspective(img, matrix, (595, 842))

plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.subplot(122)
plt.imshow(result)
plt.axis('off')
plt.show()
