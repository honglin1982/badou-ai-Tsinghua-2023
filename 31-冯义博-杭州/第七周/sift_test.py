import cv2

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
# 特征点，描述信息检测计算
keypoints, descriptor = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(img, keypoints, img, (0, 255, 127), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("sift", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
