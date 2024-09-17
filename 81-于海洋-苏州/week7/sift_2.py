# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/6/15
@author: 81-于海洋

"""
import cv2
import numpy as np

from config import Config


def draw_match(img_1, kp_1, img_2, kp_2, match):
    h1, w1 = img_1.shape[:2]
    h2, w2 = img_2.shape[:2]
    print("w1:", w1, ",w2:", w2, "maxH：", max(h1, h2))
    dest = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    dest[:h1, :w1] = img_1
    dest[:h2, w1:w1 + w2] = img_2

    h3, w3 = dest.shape[:2]
    print("w1:", h3, ",w2:", w3)

    p1 = [kp.queryIdx for kp in match]
    p2 = [kp.trainIdx for kp in match]

    post1: np.ndarray = np.int32([kp_1[p].pt for p in p1])
    post2: np.ndarray = np.int32([kp_2[p].pt for p in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(dest, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", dest)


if __name__ == '__main__':
    src_img_1 = cv2.imread(Config.LENNA)
    src_img_2 = cv2.imread(Config.LENNA_BLUR)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(src_img_1, None)
    kp2, des2 = sift.detectAndCompute(src_img_2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good_match = []
    for m, n in matches:
        if m.distance < 0.50 * n.distance:
            good_match.append(m)

    draw_match(src_img_1, kp1, src_img_2, kp2, good_match[:8])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
