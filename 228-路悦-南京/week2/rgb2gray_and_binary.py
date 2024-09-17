import cv2
import numpy as np

img = cv2.imread("lenna.png")

h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
binary_image = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
        binary_image[i, j] = 255 if img_gray[i, j] > 127 else 0

cv2.imshow("gray image", img_gray)
cv2.imshow("binary image", binary_image)

cv2.waitKey(0)  # 等待键盘按下任意键
cv2.destroyAllWindows()
