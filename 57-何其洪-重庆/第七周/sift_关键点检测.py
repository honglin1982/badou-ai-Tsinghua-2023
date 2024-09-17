# -*- coding: utf-8 -*-

import cv2

if __name__ == '__main__':
    img = cv2.imread('../resources/images/lenna.png')
    # 创建一个SIFT特征提取器对象
    sift = cv2.SIFT_create()
    # 找出关键点并计算描述符
    key_points, descriptor = sift.detectAndCompute(img, None)
    # 在图片上绘制关键点
    cv2.drawKeypoints(img, key_points, img, (51, 163, 236), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 显示图片
    cv2.imshow("sift", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
