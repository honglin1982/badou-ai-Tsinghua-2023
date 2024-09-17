import cv2
import matplotlib.pyplot as plt
import numpy as np


def drawHist(hist):
    x = [i for i in range(256)]
    plt.plot(x, hist)
    plt.hist(256, bins = 256)
    plt.title('hist')
    plt.xlabel('pixel')
    plt.ylabel('count')
    plt.show()


def showImage(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()




def histogramEqualization(img):
    d = [0 for _ in range(256)]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            d[img[i][j]] += 1
    drawHist(d)
    result = [0 for _ in range(256)]
    temp = 0
    dic = {}
    for i in range(256):
        temp += d[i]
        pixel =  int (temp * 256 / (img.shape[0] * img.shape[1])) - 1
        result[pixel] = d[i]
        dic[i] = pixel

    # 绘制均衡化直方图
    drawHist(result)

    # 均衡化后图像
    dist = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist[i][j] = dic[img[i][j]]
    showImage('equal', dist)






if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogramEqualization(gray)


