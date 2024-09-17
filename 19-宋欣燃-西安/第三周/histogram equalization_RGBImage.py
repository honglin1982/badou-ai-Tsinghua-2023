"""彩色图像直方图化
分离图像的三个通道
import numpy as np
import cv2  #导入opencv模块
image=cv2.imread("bryant.jpg")  #读取要处理的图片
B,G,R = cv2.split(image)  #分离出图片的B，R，G颜色通道
cv2.imshow("RED COMPONENT FOR ALL THREE CHANNELS",R)  #显示三通道的值都为R值时的图片
cv2.imshow("GREEN COMPONENT FOR ALL THREE CHANNELS",G)  #显示三通道的值都为G值时的图片
cv2.imshow("BLUE COMPONENT FOR ALL THREE CHANNELS",B)  #显示三通道的值都为B值时的图片
cv2.waitKey(0)  #不让程序突然结束

分离图像的三个通道,其余通道置零,注意通道的顺序是BGR:
import numpy as np
import cv2  #导入opencv模块
image=cv2.imread("bryant.jpg")  #读取要处理的图片
B,G,R = cv2.split(image)  #分离出图片的B,R,G颜色通道
zeros = np.zeros(image.shape[:2],dtype="uint8")  #创建与image相同大小的零矩阵
cv2.imshow("DISPLAY BLUE COMPONENT",cv2.merge([B,zeros,zeros]))  #显示(B,0,0)图像
cv2.imshow("DISPLAY GREEN COMPONENT",cv2.merge([zeros,G,zeros]))  #显示(0,G,0)图像
cv2.imshow("DISPLAY RED COMPONENT",cv2.merge([zeros,zeros,R]))  #显示(0,0,R)图像
cv2.waitKey(0)

将提取出来的三个通道再重新合并回去:
import numpy as np
import cv2  #导入opencv模块
image=cv2.imread("bryant.jpg")  #读取要处理的图片
B,G,R = cv2.split(image)  #分离出图片的B,R,G颜色通道
cv2.imshow("MERGE RED,GREEN AND BLUE CHANNELS",cv2.merge([B,G,R]))  #显示(B,G,R)图像
cv2.waitKey(0)
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/songxinran/Documents/GitHub/badou-AI-Tsinghua-2023/lenna.png")
cv2.imshow("RGB", img)

# 分解通道，对每一个通道均衡化
b, g, r = cv2.split(img)
# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)

bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# cv2.imshow("bH", bH)
# cv2.imshow("gH", gH)
# cv2.imshow("rH", rH)

# plt.hist(bH.ravel(), 256) # 将数组多维度拉成一维数组
# plt.hist(gH.ravel(), 256)
# plt.hist(rH.ravel(), 256)
# plt.show()

result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)

