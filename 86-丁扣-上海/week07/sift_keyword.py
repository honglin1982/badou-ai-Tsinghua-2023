import cv2

"""个人理解：
SIFT算法：就是对图像进行匹配的，通过高斯滤波模拟人类视觉的远近，形成一组高斯卷积的图层，尺寸不变，而其他组则是借用第一组的第三层图像作为底层，
再进行卷积......形成高斯金字塔=求出dog金字塔，每张dog层再进行特征点（关键点）选取，以及方向描述等，这样可以得出该图像的关键匹配点的
位置，尺度，方向，以此进行匹配。
"""

img = cv2.imread('../file/lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()  # mac m1系统无法安装python3.7版本，故只能使用高版本的opencv-python，代替cv2.xfeatures2d.SIFT_create()

# keypoints: 关键点信息，包括位置，尺度，方向信息
# descriptor: 关键点描述符，每个关键点对应128个梯度信息的特征向量
keypoints, descriptor = sift.detectAndCompute(img_gray, None)

'''
image:原始图像
keypoints:关键点信息，将其绘制在图像上
outputimage:输出图片，可以是原始图像
color:颜色设置，通过修改（b,g,r）的值，更改画笔的颜色，b=蓝色，g=绿色，r=红色。
flags:绘图功能的标识设置
'''
dst = cv2.drawKeypoints(img, keypoints=keypoints, outImage=img, color=(255, 150, 230),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift_keypoints', dst)
cv2.waitKey(20000)
cv2.destroyAllWindows()
