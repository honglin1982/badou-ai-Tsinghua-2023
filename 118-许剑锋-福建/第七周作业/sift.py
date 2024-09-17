'''
1. 生成高斯差分金字塔（DOG金字塔），尺度空间构建
2. 空间极值点检测（关键点的初步查探
3. 稳定关键点的精确定位
4. 稳定关键点方向信息分配
5。 关键点描述
6. 特征点匹配
'''

import cv2
import numpy as np

def get_key_point(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # key_points 1100 description (1100, 128)
    key_points, description = sift.detectAndCompute(gray, None)
    print(dir(key_points[0]))
    print(description.shape)
    image = cv2.drawKeypoints(image=image, outImage=image, keypoints=key_points, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))
    cv2.imshow('key_points', image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()






if __name__ == '__main__':

    image = cv2.imread('lenna.png')
    get_key_point(image)
