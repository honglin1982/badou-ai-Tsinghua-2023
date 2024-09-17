import cv2
import numpy as np
def function (img):
    height,width,channels = img.shape                   # 给定长宽和通道，image分辨率是导入的文件属性512*512
    emptyImage = np.zeros((800,800,channels),np.uint8)  #建立800*800空矩阵，案例这里固定死了，但是可以把它优化到定义函数内，便可自定义
    sh = 800 / height                                   # 按比例进行缩放或放大 高
    sw = 800 / width                                    # 按比例进行缩放或放大 宽
    for i in range (800):                               #案例中是直接遍历800*800上过每一个点以赋予i和j
        for j in range (800):
            x=int(i/sh+0.5)                             #先算x、y横纵坐标，
            y=int(j/sw+0.5)                             #int 是为了强制取整；i/sh就是这个点的位置，是放大后大小和原图的比例和；+0.5是为了向上取整的关系。
            emptyImage[i,j]=img[x,y]                    #[i,j]要知道在原图像上像素值是多少。
    return emptyImage
img = cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)