import cv2


img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)

# img = cv2.drawKeypoints(img_gray, kp, img)

img = cv2.drawKeypoints(image=img, outImage=img, keypoints=kp,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift_keypoints', img)
cv2.waitKey()
