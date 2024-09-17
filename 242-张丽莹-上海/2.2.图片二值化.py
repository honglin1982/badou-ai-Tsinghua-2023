import cv2

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img_gray.shape[:2]
for i in range(h):
    for j in range(w):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
img_binary = img_gray
print(img_binary)
print("image show binary: %s" % img_binary)
cv2.imshow("image show binary", img_binary)
cv2.waitKey()

# for循环可以改成使用np.where函数
# img_binary = np.where(img_gray >= 0.5,1,0)
