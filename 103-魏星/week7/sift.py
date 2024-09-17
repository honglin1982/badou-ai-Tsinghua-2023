import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
角点检测:
找到图像的特征点,并画在图像上
'''
def detectAndCreate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建SIFT对象
    sift = cv2.SIFT_create()
    # 在灰度图像上找关键点
    # kp = sift.detect(gray)
    kp,des = sift.detectAndCompute(gray,None)
    # 将找到的点标记到灰度图像去
    kp_image = cv2.drawKeypoints(gray, kp, None)

    return kp_image,kp,des

'''
特征匹配
'''
def get_match(des1, des2):
    bf = cv2.BFMatcher()
    #对两个匹配选择最佳的匹配
    matches = bf.knnMatch(des1, des2, k=2)
    # des1为模板图，des2为匹配图
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

'''
计算视角变换矩阵，并对右图进行变换并返回全景拼接图像
'''
def Panorama_stitching(image_right, image_left):

    _, keypoints_right, features_right = detectAndCreate(image_right)
    _, keypoints_left, features_left = detectAndCreate(image_left)
    goodMatch = get_match(features_right, features_left)

    # 当筛选项的匹配对大于4对(因为homography单应性矩阵的计算需要至少四个点)时,计算视角变换矩阵
    if len(goodMatch) > 6:
        # 获取匹配对的点坐标
        Point_coordinates_right = np.float32(
            [keypoints_right[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        Point_coordinates_left = np.float32(
            [keypoints_left[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

        # ransacReprojThreshold：将点对视为内点的最大允许重投影错误阈值(仅用于RANSAC和RHO方法时)
        # 若srcPoints和dstPoints是以像素为单位的，该参数通常设置在1到10的范围内
        ransacReprojThreshold = 4

        # cv2.findHomography():计算多个二维点对之间的最优单映射变换矩阵 H(3行x3列),使用最小均方误差或者RANSAC方法
        # 作用:利用基于RANSAC的鲁棒算法选择最优的四组配对点，再计算转换矩阵H(3*3)并返回,以便于反向投影错误率达到最小
        Homography, status = cv2.findHomography(
            Point_coordinates_right, Point_coordinates_left, cv2.RANSAC, ransacReprojThreshold)

        # cv2.warpPerspective()：透视变换函数，用于解决cv22.warpAffine()不能处理视场和图像不平行的问题
        # 作用：就是对图像进行透视变换，可保持直线不变形，但是平行线可能不再平行
        Panorama = cv2.warpPerspective(
            image_right, Homography, (image_right.shape[1] + image_left.shape[1], image_right.shape[0]))

        # 将左图加入到变换后的右图像的左端即获得最终图像
        Panorama[0:image_left.shape[0], 0:image_left.shape[1]] = image_left

        # 返回全景拼接的图像
        return Panorama

def main():
    dim = (525,350)
    img = cv2.imread("test1.png")
    img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_corner,kp1,des1 = detectAndCreate(img)
    cv2.imwrite("test1-corner.png", img_corner)

    img2 = cv2.imread("test2.png")
    img2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_corner,kp2,des2 = detectAndCreate(img2)
    cv2.imwrite("test2-corner.png", img2_corner)

    goodMatch = get_match(des1, des2)
    # drawMatches(图1，图1特征点，图2，图2特征点，图1的特征点匹配图2的特征点（所有），颜色都是默认，flags表示有几个图像)
    all_match_img = cv2.drawMatches(img, kp1, img2, kp2, goodMatch, None, flags=2)

    result_img = Panorama_stitching(img2,img)
    cv2.imwrite("result_img.png", cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

    plt.figure()

    plt.subplot(2,3,1)
    plt.title("test1-src")
    plt.imshow(img)
    plt.subplot(2,3,2)
    plt.title("test1-corner")
    plt.imshow(img_corner)

    plt.subplot(2,3,3)
    plt.title("test2-src")
    plt.imshow(img2)
    plt.subplot(2,3,4)
    plt.title("test2-corner")
    plt.imshow(img2_corner)

    plt.subplot(2,3,5)
    plt.title("all_match_img")
    plt.imshow(all_match_img)
    plt.subplot(2,3,6)
    plt.title("result_img")
    plt.imshow(result_img)

    plt.show()

if __name__ ==  '__main__':
    main()


