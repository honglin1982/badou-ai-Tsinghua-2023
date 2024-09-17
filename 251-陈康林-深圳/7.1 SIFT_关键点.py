import cv2

#读取图像并灰度化
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#创建sift算法并找到特征点
sift = cv2.xfeatures2d.SIFT_create()
kps, des = sift.detectAndCompute(gray,None)
#在原图上绘制特征点
img = cv2.drawKeypoints(image=img,keypoints=kps,outImage=img,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))
#显示图像
cv2.imshow('sift_keypoints',img)
cv2.waitKey(0) 
cv2.destroyAllWindows()