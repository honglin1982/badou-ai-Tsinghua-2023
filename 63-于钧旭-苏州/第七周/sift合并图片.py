import cv2
import numpy as np


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # 创建暴力匹配器
    matcher = cv2.DescriptorMatcher_create("BruteForce")

    # 使用 KNN 匹配
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    # 仅保留满足距离比率小于一定阈值的匹配
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当匹配对数大于4时，计算透视变换矩阵
    if len(matches) > 4:
        # 获取关键点的位置

        ptsA = np.int32([kpsA[i].pt for (_, i) in matches])
        ptsB = np.int32([kpsB[i].pt for (i, _) in matches])
        print(ptsA)
        print(ptsB)

        # 计算透视变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)

        for (x1, y1), (x2, y2) in zip(ptsA, ptsB+ (w1, 0)):
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.imshow("match", vis)

        # 返回透视变换矩阵和匹配结果
        return (matches, H)

    # 当匹配对数小于等于4时，直接退出
    return None


# 读入图片
imageA = cv2.imread("img/js2.png")
imageB = cv2.imread("img/js1.png")

# 提取 SIFT 特征
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
h1, w1 = grayA.shape[:2]
h2, w2 = grayB.shape[:2]

vis = np.zeros((max(h1, h2), w1 + w2,3), np.uint8)
vis[:h1, :w1] = imageA
vis[:h2, w1:w1 + w2] = imageB

sift = cv2.xfeatures2d.SIFT_create()
(kpsA, featuresA) = sift.detectAndCompute(grayA, None)
(kpsB, featuresB) = sift.detectAndCompute(grayB, None)

# 匹配特征点
M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0)

# 显示匹配结果
matches, H = M
result = cv2.warpPerspective(imageA, H,
                             (w1 + w2 +300, (max(h1, h2))+300))
result[0:h2, 0:w2] = imageB
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()