import cv2
import numpy as np
"""
sift流程:
1.创建sift算法
2.找出两张图的关键点
3.计算欧式距离的点
4.欧氏距离的阈值筛选
5.画图
"""

def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    """ 连线底层逻辑写法 """
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1_gray = cv2.imread("../file/jietu1.png")
img2_gray = cv2.imread("../file/jietu2.png")

# 创建sift算法
# sift = cv2.SIFT()
sift = cv2.SIFT_create()
# sift = cv2.SURF()

# 找出两张图的关键点
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 关键点匹配，得出需要进行欧式距离计算的点
bf = cv2.BFMatcher(cv2.NORM_L2)  # 特征匹配算法, NORM_L2：归一化数组的（欧几里得距离）
matches = bf.knnMatch(des1, des2, k=2)  # 第一张图中的点对应第二张图中两个特征点

# 欧式距离点的筛选
goodMatch = []
for m, n in matches:
    print(m.distance, n.distance)
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)
        print(m.queryIdx, m.trainIdx)

# 画图
# drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])  # 对图像的关键点进行连线操作。
dst = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, goodMatch[100:200], None, flags=2)  # 可以用cv2.drawMatches进行连线

cv2.imshow('sift_keypoints', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
