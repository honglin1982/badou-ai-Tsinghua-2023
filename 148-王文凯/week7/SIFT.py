import cv2
import matplotlib.pyplot as plt
import numpy as np

# 特征提取与描述，获取图像特征点
# 参考文档 https://opencv.apachecn.org/4.0.0/5.4-tutorial_py_sift_intro/
def detect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    # kp 特征点列表 des 描述子 np数组 特征点数目 * 128
    kp, des = sift.detectAndCompute(img_gray, None)
    # kp_img 特征图像
    kp_img = cv2.drawKeypoints(image=img_gray, outImage=img_gray, keypoints=kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 0))
    return kp, des, kp_img

# 图像特征匹配
# 参考文档 https://opencv.apachecn.org/4.0.0/5.9-tutorial_py_matcher/#_3
def img_match(des_1, des_2):
    # 两个可选参数
    # normType 默认值为cv2.NORM_L2 适用于SIFT 和 SURF
    # crossCheck 默认值为false 如果为 True，则 Matcher 仅返回具有值(i,j)的匹配,使得集合 A 中的第 i 个描述子具有集合 B 中的第 j 个描述子作为最佳匹配
    # BFMatcher.match() 返回最佳匹配 BFMatcher.knnMatch() 返回 k 个最佳匹配，其中 k 由用户指定。
    bf = cv2.BFMatcher()
    # DMatch.distance - 描述子之间的距离。越低越好
    # DMatch.trainIdx - 目标图像中描述子的索引
    # DMatch.queryIdx - 查询图像中描述子的索引
    # DMatch.imgIdx - 目标图像的索引
    matches = bf.knnMatch(des_1, des_2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# 全景拼接
# 参考文档 https://opencv.apachecn.org/4.0.0/5.10-tutorial_py_feature_homography/
def panorama_montage(img_2, img_1):
    img_1_kp, img_1_des, _ = detect(img_1)
    img_2_kp, img_2_des, _ = detect(img_2)
    good_matches = img_match(img_1_des, img_2_des)
    if len(good_matches) > 10:
        src_pts = np.float32([img_1_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([img_2_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        res_img = cv2.warpPerspective(img_1, m, (img_1.shape[1] + img_2.shape[1], img_1.shape[0]))
        res_img[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
        return res_img

def main():
    img_1 = cv2.cvtColor(cv2.imread('./img/img_1.jpg'), cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(cv2.imread('./img/img_2.jpg'), cv2.COLOR_BGR2RGB)

    img_1_kp, img_1_des, img_1_kp_img = detect(img_1)
    img_2_kp, img_2_des, img_2_kp_img = detect(img_2)

    good_matches = img_match(img_1_des, img_2_des)
    img_match_res = cv2.drawMatches(img_1, img_1_kp, img_2, img_2_kp, good_matches, None, flags=2)

    res_img = panorama_montage(img_1, img_2)

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.title("img_1")
    plt.imshow(img_1)
    plt.subplot(3, 2, 2)
    plt.title("img_1_kp_img")
    plt.imshow(img_1_kp_img)
    plt.subplot(3, 2, 3)
    plt.title("img_2")
    plt.imshow(img_2)
    plt.subplot(3, 2, 4)
    plt.title("img_2_kp_img")
    plt.imshow(img_2_kp_img)
    plt.subplot(3, 2, 5)
    plt.title('img_match_res')
    plt.imshow(img_match_res)
    plt.subplot(3, 2, 6)
    plt.title('res_img')
    plt.imshow(res_img)
    plt.show()

if __name__ == "__main__":
    main()
