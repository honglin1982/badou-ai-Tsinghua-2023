import cv2
import numpy as np


# 输出关键点图像
def sift_feature():
    img = cv2.imread("image/lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建 sift 特征提取器
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述符
    keypoints, descriptors = sift.detectAndCompute(img_gray, mask=None)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
    output_img = np.copy(img)
    cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=output_img, color=(51, 163, 236),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Number of keypoints:", len(keypoints))
    print("Descriptor shape:", descriptors.shape)

    # 显示关键点图像
    cv2.imshow('sift_keypoints', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像关键点比对
def sift_map():
    # 读取图像
    img_1 = cv2.imread("image/iphone1.png")
    img_2 = cv2.imread("image/iphone2.png")
    print("查询、训练图像信息:", img_1.shape, img_2.shape)
    # 创建 SIFT 特征提取器
    sift = cv2.SIFT_create()
    # 检测关键点并计算其描述子
    kp1, des1 = sift.detectAndCompute(img_1, mask=None)
    kp2, des2 = sift.detectAndCompute(img_2, mask=None)
    print("查询图像的关键点和描述子:", len(kp1), des1.shape)
    print("训练图像的关键点和描述子:", len(kp2), des2.shape)
    # 创建Brute-Force匹配器：用于在给定的特征描述子集合中进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_L2)   # cv2.NORM_L2：使用欧氏距离
    # 进行特征匹配: knnMatch 对两组特征描述子进行k最近邻匹配，返回每个查询描述子的k个最佳匹配结果
    k_matches = bf.knnMatch(des1, des2, k=2)

    good_match = []  # 保存好的匹配结果
    threshold_value = 0.50  # 设定阈值为多少倍的 n.distance
    print(f"设定阈值为: {threshold_value}")
    for m, n in k_matches:
        if m.distance < threshold_value*n.distance:   # .distance取特征点之间的距离
            good_match.append(m)  # 满足条件添加到好的匹配列表中
    print(f"关键点良好的匹配结果数量: {len(good_match)}")

    # 获取图像宽高
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    # 创建用于关键点映射的基底图
    map_img = np.zeros((max(h1, h2), w1+w2, 3), dtype=np.uint8)
    # 将两张图片映射进该图中
    map_img[:h1, :w1] = img_1
    map_img[:h2, w1:w1+w2] = img_2
    """
    m.queryIdx：匹配特征点在查询图像中的索引。
    m.trainIdx：匹配特征点在训练图像中的索引。
    m.imgIdx：匹配特征点所属图像的索引（在多图像匹配时使用）。
    img_1 为查询图像， img——2 为训练图像
    """
    index_1 = [kp.queryIdx for kp in good_match]
    index_2 = [kp.trainIdx for kp in good_match]
    # 通过索引获取关键点的位置信息
    coordinates1 = np.int32([kp1[i].pt for i in index_1])   # .pt 获取位置信息
    # 因为在基底图中，图2的开始位置是 （w1， 0），所以每个关键点都要加上该开始位置
    coordinates2 = np.int32([kp2[i].pt for i in index_2]) + (w1, 0)
    # 绘制直线
    for (x1, y1), (x2, y2) in zip(coordinates1, coordinates2):
        cv2.line(img=map_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255))
    # 创建图像窗口
    cv2.namedWindow("match_img", flags=cv2.WINDOW_NORMAL)  # 窗口大小可以调整
    cv2.namedWindow("match_img", cv2.WINDOW_AUTOSIZE)  # 大小自动调整
    cv2.imshow("match_img", map_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sift_feature()
    sift_map()
