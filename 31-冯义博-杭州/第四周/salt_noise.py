import cv2
import random

"""
单通道
"""
# img = cv2.imread("lenna.png")
# n_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# w, h = n_img.shape[:2]
# num = int(w * h * 0.2)
# for i in range(num):
#     y = random.randint(0, h - 1)
#     x = random.randint(0, w - 1)
#     if random.random() > 0.5:
#         n_img[y, x] = 255
#     else:
#         n_img[y, x] = 0

"""
三通道
"""
n_img = cv2.imread("lenna.png")
w, h, c = n_img.shape[:3]
num = int(w * h * 0.2)
for i in range(c):
    for j in range(num):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        if random.random() > 0.5:
            n_img[y, x][i] = 255
        else:
            n_img[y, x][i] = 0

# cv2.imshow("source", cv2.cvtColor(cv2.imread("lenna.png"), cv2.COLOR_BGR2GRAY))
cv2.imshow("source", cv2.imread("lenna.png"))
cv2.imshow("salt_noise", n_img)
cv2.waitKey(0)
