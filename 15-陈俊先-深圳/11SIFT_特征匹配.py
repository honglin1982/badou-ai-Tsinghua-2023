import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, good_match):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray  # 将原始图像 img1_gray 显示在可视化图像的左侧，以便与另一张图像进行对比和匹配可视化
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in good_match]  # .queryIdx 属性表示查询图像中关键点的索引
    p2 = [kpp.trainIdx for kpp in good_match]  # .trainIdx 属性用于表示每个查询关键点所匹配的训练关键点的索引

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)
    '''
    在计算机视觉中，关键点（keypoint）是图像中具有特定特征的点，通常用于表示图像中的显著位置或感兴趣区域。关键点可以通过特征检测算法（如SIFT、SURF、ORB等）在图像中自动提取出来。

    关键点对象通常具有一些属性，例如坐标、尺度、方向、响应值等，这些属性用于描述关键点的特征信息。其中，.pt 是关键点对象的一个属性，表示关键点的坐标信息。

    在给定的代码中，kp2[pp] 表示索引为 pp 的关键点对象，通过访问其 .pt 属性，可以获取该关键点的坐标信息。这样，通过列表推导式 [kp2[pp].pt for pp in p2]，我们可以从关键点列表 kp2 中提取出与 p2 列表中索引对应的关键点的坐标信息，并生成一个新的列表。

    最后，通过 np.int32 将列表中的坐标信息转换为整数类型，并使用 (w1, 0) 进行偏移，实现在图像中绘制匹配线段时的位置调整。
    '''

    # 绘制连接线
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.namedWindow('match', cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL，表示窗口可以调整大小
    cv2.imshow('match', vis)


img1_gray = cv2.imread('iphone1.png')
img2_gray = cv2.imread('iphone2.png')

sift = cv2.SIFT_create()    # 3.4以前版本用sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_gray, None)  # kp1是关键点（KeyPoint）列表，表示图像1中检测到的特征点，而 des1 是描述符（Descriptor）列表，表示图像1中每个特征点的特征描述
kp2, des2 = sift.detectAndCompute(img2_gray, None)

bf = cv2.BFMatcher(cv2.NORM_L2)  # cv2.BFMatcher是一个Brute-Force匹配器，用于在特征描述符之间进行匹配。它会对第一个特征描述符中的每个特征点，与第二个特征描述符中的所有特征点进行距离计算，然后返回最佳匹配
# cv2.NORM_L2代表用欧氏距离来进行计算
matches = bf.knnMatch(des1, des2, k=2)

good_match = []
for m, n in matches:
    if m.distance < 0.50*n.distance:    # 这里采用最佳特征点的距离与次最佳特征点的距离进行比较，存疑？
        good_match.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, good_match[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()
