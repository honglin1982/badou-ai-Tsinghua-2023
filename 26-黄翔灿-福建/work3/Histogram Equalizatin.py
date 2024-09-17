import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

def His_equl(img,Len=256):
    src_h, src_w = img.shape[0:2]
    Si = src_h * src_w
    sum_h = 0
    #计算图像存在每个灰度值的像素点数量
    hist = cv2.calcHist([img],[0],None,[256], [0,256])
    #归一化计算,灰度值像素点的数量
    hist[0:256] = hist[0:256] / Si
    sum_Pi = np.zeros(hist.shape)
    #计算前i个灰度值分布概率的累加
    for i in range(256):
        sum_Pi[i] = sum(hist[0:i+1])
    #sun_Pi * 256 -1
    equal_hist = np.zeros(sum_Pi.shape)
    for i in range(256):
        equal_hist[i] = int(Len * sum_Pi[i] - 0.5)
    #创建新图像
    Hist_img = img.copy()
    for i in range(src_h):
        for j in range(src_w):
         Hist_img[i,j] = equal_hist[img[i,j]][:1]
    #变化后的直方图
    dst_hist = cv2.calcHist([Hist_img],[0],None,[256], [0,256])
    dst_hist[0:255] = dst_hist[0:255] / Si

    return [Hist_img,dst_hist]
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst_img, hist = His_equl(img_1)[0],His_equl(img_1)[1]
    cv2.imshow("reverse", dst_img)
    plt.figure()
    plt.hist(dst_img.ravel(), 256)
    plt.show()
    cv2.waitKey()