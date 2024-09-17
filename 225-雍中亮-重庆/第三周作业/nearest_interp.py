import cv2
import numpy as np
def function(img):
    height,width,channels=img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage
# 有对应的最邻近插值算法
# cv2.resize(img,(800,800,c),near/bin)
# 上述的选择采用near算法还是采用bin算法。
img=cv2.imread("lenna.png")
zoom=function(img)

print(zoom)
cv2.imshow("nearest",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
