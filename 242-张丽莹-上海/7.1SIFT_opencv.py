# SIFT完全调接口版

import cv2

# 读取图片
img1 = cv2.imread('mountain1.png')
img2 = cv2.imread('mountain2.png')

# 创建特征提取器对象sift，xfeatures2d是OpenCV的特征提取器模块
# SIFT_create()是特征提取器对象的构造函数，会返回一个SIFT特征提取器对象
# 所以这俩是连着用的，就记住吧！
sift = cv2.xfeatures2d.SIFT_create()

# 特征提取器对象sift在img1中检测出关键点(keypoints)和描述符(descriptors)
# detectAndCompute第一个参数是待提取图像，第二个参数是可选的掩码，none是在整个图的意思
# 返回值：元组：关键点列表kp + 特征描述符矩阵des
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 创建Brute-Force匹配器对象bf，用于在两个特征描述集合中寻找最佳匹配
# cv2.BFMatcher没有参数是默认基于欧氏距离来计算两个特征描述符的距离，并使用暴力搜索算法来查找匹配
bf = cv2.BFMatcher()

# 用Brute-Force匹配器对象的bf方法来匹配描述符，获得列表matches
# 每个匹配项包含两个特征描述符的索引及它们之间距离
matches = bf.match(des1, des2)

# 按照距离排序：对matches列表内容排序
# 关键字使用lambda匿名函数，返回匹配项x的距离值x.distance作为排序值
matches = sorted(matches, key=lambda x: x.distance)

# 可视化匹配结果：在输入图像img1和img2中绘制最佳匹配的前10个项，生成输出图像img3
# 注意这里参数是图像及其关键点列表，匹配项数量。flags=2表示绘制所有匹配项，包括不好的
# 该函数绘制的方式为：两张图里的匹配项用线段连起来
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

# 显示图片
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
