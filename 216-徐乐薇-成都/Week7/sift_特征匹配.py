import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np

# sift特征匹配
# 1. 读取图片
# 2. 创建sift对象
# 3. 计算关键点和描述子
# 4. 创建BFMatcher对象
# 5. knnMatch匹配
# 6. 画出匹配线

# 画出匹配线
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch): #定义一个函数，用来画出匹配线,kp表示关键点，goodMatch表示匹配的点
    # 获取两张图的高和宽
    h1, w1 = img1_gray.shape[:2] #获取第一张图片的高和宽
    h2, w2 = img2_gray.shape[:2] #获取第二张图片的高和宽

    # 创建一个新图，包括两张图片的大小
    vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8) #创建一个新图，包括两张图片的大小, 3表示RGB, np.uint8表示8位无符号整型
    vis[:h1, :w1] = img1_gray #将第一张图片放到左上角
    vis[:h2, w1:w1+w2] = img2_gray #将第二张图片放到右上角

    # 获取两张图的关键点索引
    p1 = [kpp.queryIdx for kpp in goodMatch] #获取第一张图片的关键点的索引,kpp.queryIdx表示第一张图片的索引,kpp.trainIdx表示第二张图片的索引,kpp.distance表示两个关键点之间的距离
    p2 = [kpp.trainIdx for kpp in goodMatch] #获取第二张图片的关键点的索引

    # 获取两张图的关键点坐标
    post1 = np.int32([kp1[pp].pt for pp in p1]) #获取第一张图片的关键点的坐标
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0) #获取第二张图片的关键点的坐标，并且加上第一张图片的宽度，因为第二张图片要放在第一张图片的右边

    # 画出匹配线
    for (x1, y1), (x2, y2) in zip(post1, post2): #zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255)) #画出匹配线，(x1, y1)表示第一张图片的关键点坐标，(x2, y2)表示第二张图片的关键点坐标，(0, 0, 255)表示红色

    # 显示图片
    cv2.namedWindow("match", cv2.WINDOW_NORMAL) #创建一个窗口，用来显示图片, cv2.WINDOW_NORMAL表示可以改变窗口大小
    cv2.imshow("match", vis) #显示图片, match表示窗口的名字，vis表示要显示的图片

# 1. 读取图片
img1_gray = cv2.imread("iphone1.png") #读取第一张图片
img2_gray = cv2.imread("iphone2.png") #读取第二张图片

# 2. 创建sift对象
# sift = cv2.SIFT() #创建sift对象
sift = cv2.xfeatures2d.SIFT_create() #创建sift对象

# 3. 计算关键点和描述子
kp1, des1 = sift.detectAndCompute(img1_gray, None) #计算第一张图片的关键点和描述子,des1表示描述子 kp1表示关键点 kp1是一个列表，列表中的每个元素是一个KeyPoint对象，KeyPoint对象包含四个属性，分别是angle, class_id, octave, pt
kp2, des2 = sift.detectAndCompute(img2_gray, None) #计算第二张图片的关键点和描述子

# 4. 创建BFMatcher对象,并进行匹配
# BFMatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2) #创建BFMatcher对象 cv2.NORM_L2表示欧式距离
matches = bf.knnMatch(des1, des2, k=2) #进行匹配，k=2表示每个关键点返回两个最佳匹配,bf.knnMatch()返回的是一个列表，列表中的每个元素是一个列表，列表中的每个元素是一个DMatch对象，DMatch对象包含三个属性，分别是queryIdx, trainIdx, distance

# 5. knnMatch匹配
goodMatch = [] #创建一个列表，用来存放匹配的点
for m, n in matches: #遍历所有的匹配点，m表示第一张图片的匹配点，n表示第二张图片的匹配点,m.distance表示第一张图片的匹配点与第二张图片的匹配点之间的距离
    if m.distance < 0.4 * n.distance: #如果第一张图片的匹配点的距离小于0.4倍的第二张图片的匹配点的距离，就认为是好的匹配点，因为第一张图片的匹配点与第二张图片的匹配点越接近，两个匹配点之间的距离就越小
        goodMatch.append(m) #将第一张图片的匹配点添加到goodMatch列表中

# 6. 画出匹配线
drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20]) #画出匹配线

# 7. 显示图片
cv2.waitKey(0) #等待按键按下
cv2.destroyAllWindows() #销毁所有窗口

# 8. 保存图片
cv2.imwrite("match.png", vis) #保存图片

