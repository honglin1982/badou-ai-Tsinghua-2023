# -*- coding: utf-8 -*-

import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png')
    # 旋转90°
    img2 = np.rot90(img)
    # 创建一个SIFT特征提取器对象
    sift = cv2.SIFT_create()
    # 找出关键点并计算描述符
    key_points, descriptor = sift.detectAndCompute(img, None)
    key_points2, descriptor2 = sift.detectAndCompute(img2, None)
    # 创建匹配器
    bf_matcher = cv2.BFMatcher_create(cv2.NORM_L2)
    # 描述匹配
    matchers = bf_matcher.knnMatch(descriptor, descriptor2, k=2)
    # 筛选匹配较好的点
    good_match = []
    for m, n in matchers:
        if m.distance < 0.1 * n.distance:
            good_match.append([m])
    print(len(good_match))
    # 图片上绘制关联
    result = cv2.drawMatchesKnn(img, key_points, img2, key_points2, good_match, None, flags=2)
    cv2.imshow("knnMatch", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
