import cv2
import numpy as np

def drawKnnMatches(img1, kp1, img2, kp2, good_matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 创建一个能容纳两个图像并排的空间，方便观看
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
 
    # good_matches里的元素是cv2.DMatch对象，包含了在查询图像和参考图像中的索引
    # 即queryIdx和trainIdx
    # 取出所有索引
    p1 = [i.queryIdx for i in good_matches]
    p2 = [i.trainIdx for i in good_matches] 
 
    '''
    遍历 p1 列表中的每个索引值 i；
    从 kp1 列表中获取索引为 i 的关键点，并提取其在图像中的坐标信息 pt；
    将所有成功匹配的特征点在查询图像中的坐标信息存储到一个列表中；
    将该列表转换为一个 NumPy 数组，并将其中的浮点数坐标值转换为整型坐标值。

    需注意post2为参考图像中的坐标信息，我们为了在vis里有更好的观看，所以要
    平移坐标
    '''
    post1 = np.int32([kp1[i].pt for i in p1])
    post2 = np.int32([kp2[i].pt for i in p2]) + (w1, 0)
 
    # print(list(zip(post1, post2)))
    for (x1, y1), (x2, y2) in zip(post1, post2):
        '''
        cv2.line表示在vis上绘制查询图像某个特征点（x1，y1）到参考图像中的
        某个特征点（x2，y2）的线段
        (0,0,255) 表示绘制的线段颜色，这里使用的是 RGB 颜色空间中的红色
        '''
        cv2.line(vis, (x1, y1), (x2, y2), (0,0,255))
 
    # cv2.namedWindow(windom_name, type)
    # 创建一个名为match的窗口，cv2.WINDOW_NORMAL表示这个窗口可以任意调节大小
    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)
    cv2.waitKey(0)


img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')

'''sift检测灰度图关键点和描述'''
# 创建sift对象
sift = cv2.xfeatures2d.SIFT_create()
# 检测关键点和描述
'''
kp, des = sift.detectAndCompute(image, mask, useProvidedKeypoints=False)
image：需要进行特征提取的源图像，可以是灰度图像或彩色图像，
       数据类型为 numpy.ndarray，通常使用 cv2.imread 函数读取；
mask：掩码图像，只有与掩码图像对应位置为白色（255）的像素点才会被考虑，其他像素点将被忽略，
      数据类型同样为 numpy.ndarray；
useProvidedKeypoints：可选参数，布尔类型，是否使用用户提供的关键点，如果为 True，
      则需要在 kp 参数中提供关键点，否则函数将自动检测关键点，并返回关键点和特征描述符。
'''
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

'''特征匹配'''
'''
bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    normType - 可选参数，用于指定特征描述符之间的距离度量方式，有以下几种取值：
                cv2.NORM_L1：使用 L1 距离（绝对值差）；
                cv2.NORM_L2：使用 L2 距离（欧式距离）；(默认值)
                cv2.NORM_HAMMING：使用汉明距离（二进制字符串中不同位置的个数）；
                cv2.NORM_HAMMING2：使用双边汉明距离（二进制字符串中不同位置的个数加权后的结果）。
    crossCheck - 可选参数，布尔类型，用于指定是否启用交叉检查，如果为 True，则只有当第一个特征描述符
                与第二个特征描述符的距离小于第二个特征描述符与第一个特征描述符的距离时，才会被认为是
                匹配成功的，否则将被视为匹配失败。
'''
# 创建一个bf对象
bf = cv2.BFMatcher()
'''
bf.knnMatch(des1, des2, k=2) 是 cv2.BFMatcher 类的一个函数，用于进行双向最近邻匹配。
它与 bf.match(des1, des2) 的区别在于，knnMatch 函数返回每个查询图像中的特征点在参考
图像中的两个最佳匹配点。其中，k 参数用于指定最近邻和次近邻匹配点的个数，通常为 2。
knnMatch 函数的返回值是一个列表，其中每个元素代表查询图像中的一个特征点及其和参考
图像中两个最佳匹配点的匹配结果
'''
matches = bf.knnMatch(des1, des2, k=2)

# 由于 k设置的2，所有每个点匹配的结果有两个点，自定义一个阈值来筛选好的匹配点，
# 同过以下方式筛选

good_matches = []
for m, n in matches:
    if m.distance < 0.39*n.distance:
        good_matches.append(m)

'''画图'''
drawKnnMatches(img1, kp1, img2, kp2, good_matches[:20])