import cv2
import numpy as np

def DrawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch):

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    vis_h = max(h1,h2)
    vis = np.zeros((vis_h,w1+w2,3),np.uint8)
    vis[:h1,:w1] = img1
    vis[:h2,w1:w1+w2] = img2

    ''''
    1. "goodMatch"是一组从两个图像中提取的关键点（kps）之间的匹配列表。 
    2. 代码使用列表推导式从“goodMatch”中的每个关键点对（kpp）中提取查询索引（queryIdx），并将其存储在名为“p1”的新列表中。 
    3. 类似地，另一个列表推导式用于从“goodMatch”中的每个关键点对（kpp）中提取训练索引（trainIdx），并将其存储在名为“p2”的新列表中。 
    4. 最终得到的列表“p1”和“p2”包含了在两个图像之间匹配的关键点的索引。
    '''
    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]
    '''
    创建了两个数组post1和post2，分别包含第一张和第二张图片中匹配关键点的像素坐标。
    post2数组向右移动了第一张图片的宽度，以便在第一张图片的右侧绘制第二张图片中的对应点。
    在OpenCV中，关键点对象（KeyPoint）包含许多属性，其中之一是像素坐标（pt），它表示关键点在图像中的位置。
    因此，kp1[pp].pt表示在第一张图像中第pp个关键点的像素坐标。
    '''
    postion1 = np.int32([kp1[pp].pt for pp in p1])
    postion2 = np.int32([kp2[pp].pt for pp in p2]) + (w1,0)

    '''
    遍历post1和post2中的每对对应点，这些点分别包含第一张和第二张图片中匹配关键点的像素坐标。
    对于每对点，代码提取它们在两张图片中的x和y坐标，
    并使用cv2.line()函数在可视化图像“vis”上绘制一条连接它们的红色线条。
    其中，(0,0,255)参数指定了线条的颜色，这里是红色（蓝色=0，绿色=0，红色=255）。 
    在这段代码中，zip()函数将post1和post2中的元素一一对应地打包成元组，
    然后返回一个由这些元组组成的迭代器。在for循环中，使用两个元组解包语法，
    将每个元组中的x和y坐标分配给(x1, y1)和(x2, y2)变量。这样，代码就可以遍历每对对应点，
    将它们的坐标传递给cv2.line()函数，绘制连接它们的红色线条。
    '''
    for (x1,y1),(x2,y2) in zip(postion1,postion2):
        cv2.line(vis,(x1,y1),(x2,y2),color=(0,0,255))
    cv2.drawKeypoints(image=vis[:h1,:w1],keypoints=kp1,outImage=vis[:h1,:w1],
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(53,163,236))
    cv2.drawKeypoints(image=vis[:h2,w1:w1+w2],keypoints=kp2,outImage=vis[:h2,w1:w1+w2],
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(53,163,236))
    cv2.namedWindow('match',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('match',vis)


#读取图像
img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')

#创建sift算法
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

# BFmatcher with default parms
'''
使用OpenCV库中的BFMatcher函数创建了一个Brute-Force Matcher对象bf，并使用cv2.NORM_L2作为距离度量方式。
然后，使用des1和des2作为输入，使用knnMatch函数进行特征匹配，其中k=2表示对于每个查询特征点，
返回最匹配的两个特征点
'''
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1,des2,k=2)
'''
 从特征匹配结果matches中筛选出距离比较接近的匹配点，将这些匹配点存储在goodMatch列表中。
 具体来说，代码使用一个for循环遍历matches中的每个匹配对m和n，然后判断m的距离是否小于0.5倍的n的距离。
 如果是，则将m加入到goodMatch列表中。这个距离比较接近的判断条件是一种经验值，可以根据实际情况进行调整。
'''
goodMatch = []
for m,n in matches:
    if m.distance < 0.35 * n.distance:
        goodMatch.append(m)

DrawMatchesKnn_cv2(img1,kp1,img2,kp2,goodMatch)

cv2.waitKey(0)
cv2.destroyAllWindows()